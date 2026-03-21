/*
RMSNorm (Root Mean Square Layer Normalization) CUDA kernels for LLaMA.
RMSNorm differs from LayerNorm in two ways:
  1. No mean subtraction (no "centering")
  2. No bias term (only a learnable scale weight)
Formula: out = (x / rms(x)) * weight,  rms(x) = sqrt(mean(x^2) + eps)

References:
  - https://arxiv.org/abs/1910.07467  (Zhang & Sennrich, 2019)
  - LLaMA model code from Meta
*/
#ifndef RMSNORM_CUH
#define RMSNORM_CUH

#include <assert.h>
#include "cuda_utils.cuh"
#include "cuda_common.h"
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
#if defined(SF16_TRUE_FORWARD)
#include "q131_common.cuh"
#endif
#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
#endif

__device__ __forceinline__ float quantize_rmsnorm_backward(float x) {
    // Backward runs in native BF16 precision -- no Q115/Q131 simulation.
    // "Q1.15 simulated inference with non-sf16 backprop" (cuda_common.h).
    return x;
}

// ----------------------------------------------------------------------------
// CUDA kernel: RMSNorm forward
// Computes:  out[b,t,:] = x[b,t,:] / rms(x[b,t,:]) * w[:]
// where rms(x) = sqrt(mean(x^2) + eps)
// Stores rstd (reciprocal std) for reuse in backward pass.
// One block per (b*t) position; each block handles the full C-dim.

__global__ void rmsnorm_forward_kernel(
    floatX *__restrict__ out,   // (B*T, C)
    float  *__restrict__ rstd,  // (B*T,) — stored for backward
    const floatX *__restrict__ inp,  // (B*T, C)
    const floatX *__restrict__ weight, // (C,)
    int C, float eps)
{
    // Each block processes one (b,t) row of length C.
    int bt = blockIdx.x;
    const floatX *x = inp    + (size_t)bt * C;
    floatX       *o = out    + (size_t)bt * C;

    // Compute sum of squares using warp-level reduction.
    float thread_ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = (float)__ldcs(&x[i]);
        thread_ss += xi * xi;
    }
    float block_ss = blockReduce<warpReduceSum>(thread_ss);

    // Broadcast rstd from thread 0.
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        float rms = sqrtf(block_ss / (float)C + eps);
        s_rstd = 1.0f / rms;
        if (rstd != nullptr) rstd[bt] = s_rstd;
    }
    __syncthreads();

    float r = s_rstd;

    // Normalize and scale.
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = (float)__ldcs(&x[i]);
        float wi = (float)__ldcs(&weight[i]);
        float oi = xi * r * wi;
#if defined(ENABLE_Q115)
        oi = simulate_q115(oi);
#endif
        __stcs(&o[i], (floatX)oi);
    }
}

// Launcher for RMSNorm forward.
void rmsnorm_forward(floatX *out, float *rstd,
                     const floatX *inp, const floatX *weight,
                     int B, int T, int C, float eps,
                     cudaStream_t stream)
{
    int BT = B * T;
    // Use up to 512 threads; the kernel does a full-C sweep with strided loops.
    int block_size = min(512, C);
    // Round block_size to nearest multiple of 32.
    block_size = CEIL_DIV(block_size, 32) * 32;
    rmsnorm_forward_kernel<<<BT, block_size, 0, stream>>>(
        out, rstd, inp, weight, C, eps);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// CUDA kernel: RMSNorm backward
// Given upstream gradient dout (B*T, C), input x (B*T, C), weight w (C),
// and rstd (B*T), compute:
//   dinp  += (1/rms) * w * (dout - x * rstd^2 * dot(dout*x, w) / C)   [broadcast correction]
//   dweight += sum over (B*T) of rstd * x * dout
//
// Simplified form (equivalent to standard RMSNorm backward):
//   Let y = x * rstd  (normalised pre-scale)
//   dinp += rstd * w * dout  - rstd^3 * x * sum_i(w_i * dout_i * x_i) / C
//   dweight += sum_bt(y * dout)

__global__ void rmsnorm_backward_dinp_kernel(
    floatX *__restrict__ dinp,      // (B*T, C) — += here
    const floatX *__restrict__ dout,   // (B*T, C)
    const floatX *__restrict__ inp,    // (B*T, C)
    const floatX *__restrict__ weight, // (C,)
    const float  *__restrict__ rstd,   // (B*T,)
    int C, int BT)
{
    int bt = blockIdx.x;
    if (bt >= BT) return;

    const floatX *x  = inp  + (size_t)bt * C;
    const floatX *dy = dout + (size_t)bt * C;
    floatX       *dx = dinp + (size_t)bt * C;
    float r = rstd[bt];

    // dot(dout * weight, x) for this row
    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float dyi = (float)__ldcs(&dy[i]);
        float wi  = (float)__ldcs(&weight[i]);
        float xi  = (float)__ldcs(&x[i]);
        thread_dot += dyi * wi * xi;
    }
    float block_dot = blockReduce<warpReduceSum>(thread_dot);
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = block_dot;
    __syncthreads();

    float dot_val = s_dot;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float dyi = (float)__ldcs(&dy[i]);
        float wi  = (float)__ldcs(&weight[i]);
        float xi  = (float)__ldcs(&x[i]);
        float dxi = r * wi * dyi - r * r * r * xi * dot_val / (float)C;
        float prev_dxi = (float)__ldcs(&dx[i]);
        float summed = quantize_rmsnorm_backward(prev_dxi + dxi);
        __stcs(&dx[i], (floatX)summed);
    }
}

__global__ void rmsnorm_backward_dweight_kernel(
    floatX *__restrict__ dweight,      // (C,) +=
    const floatX *__restrict__ dout,   // (B*T, C)
    const floatX *__restrict__ inp,    // (B*T, C)
    const float  *__restrict__ rstd,   // (B*T,)
    int BT, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < C; i += stride) {
        float acc = 0.0f;
        for (int bt = 0; bt < BT; bt++) {
            float xi = (float)__ldcs(&inp[(size_t)bt * C + i]);
            float dyi = (float)__ldcs(&dout[(size_t)bt * C + i]);
            acc += rstd[bt] * xi * dyi;
        }
        float prev = (float)__ldcs(&dweight[i]);
        float summed = quantize_rmsnorm_backward(prev + acc);
        __stcs(&dweight[i], (floatX)summed);
    }
}

// Launcher for RMSNorm backward.
void rmsnorm_backward(floatX *dinp, floatX *dweight, float *scratch,
                      const floatX *dout, const floatX *inp,
                      const floatX *weight, const float *rstd,
                      int B, int T, int C,
                      cudaStream_t stream)
{
    (void)scratch;
    int BT = B * T;
    int block_size = min(512, C);
    block_size = CEIL_DIV(block_size, 32) * 32;

    // 1) dinput: one block per token row, no cross-block races
    rmsnorm_backward_dinp_kernel<<<BT, block_size, 0, stream>>>(
        dinp, dout, inp, weight, rstd, C, BT);
    cudaCheck(cudaGetLastError());

    // 2) dweight: one thread per channel with full BT reduction
    const int dw_block = 256;
    unsigned int dw_grid = (unsigned int)((C + dw_block - 1) / dw_block);
    if (dw_grid > 65535u) dw_grid = 65535u;
    if (dw_grid == 0u) dw_grid = 1u;
    rmsnorm_backward_dweight_kernel<<<dw_grid, dw_block, 0, stream>>>(
        dweight, dout, inp, rstd, BT, C);
    cudaCheck(cudaGetLastError());
}

#endif // RMSNORM_CUH
