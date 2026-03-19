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

__global__ void rmsnorm_backward_kernel(
    floatX *__restrict__ dinp,      // (B*T, C) — += here
    floatX *__restrict__ dweight,   // (C,)      — += here (per block, then atomic)
    float  *__restrict__ scratch,   // (B*T) scratch for partial sums (unused, reserved)
    const floatX *__restrict__ dout,   // (B*T, C)
    const floatX *__restrict__ inp,    // (B*T, C)
    const floatX *__restrict__ weight, // (C,)
    const float  *__restrict__ rstd,   // (B*T,)
    int B, int T, int C)
{
    // Shared memory for dweight accumulation.
    extern __shared__ float sh_dw[];  // [C] — one float per channel, zero-filled

    int bt = blockIdx.x;
    const floatX *x  = inp    + (size_t)bt * C;
    const floatX *dy = dout   + (size_t)bt * C;
    floatX       *dx = dinp   + (size_t)bt * C;
    float r = rstd[bt];

    // --- Step 1: Compute dot(dout * weight, x) for this row ---
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

    // --- Step 2: Compute dinp and accumulate dweight ---
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float dyi = (float)__ldcs(&dy[i]);
        float wi  = (float)__ldcs(&weight[i]);
        float xi  = (float)__ldcs(&x[i]);

        // dinp contribution for this row.
        float dxi = r * wi * dyi - r * r * r * xi * dot_val / (float)C;
        float prev_dxi = (float)__ldcs(&dx[i]);
        __stcs(&dx[i], (floatX)(prev_dxi + dxi));

        // Accumulate dweight in shared mem (one entry per channel).
        sh_dw[i] += r * xi * dyi;
    }
    __syncthreads();

    // --- Step 3: Atomic-add shared dweight into global dweight ---
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float prev = (float)__ldcs(&dweight[i]);
        __stcs(&dweight[i], (floatX)(prev + sh_dw[i]));
    }
}

// Launcher for RMSNorm backward.
void rmsnorm_backward(floatX *dinp, floatX *dweight, float *scratch,
                      const floatX *dout, const floatX *inp,
                      const floatX *weight, const float *rstd,
                      int B, int T, int C,
                      cudaStream_t stream)
{
    int BT = B * T;
    int block_size = min(512, C);
    block_size = CEIL_DIV(block_size, 32) * 32;
    size_t shared_bytes = (size_t)C * sizeof(float);
    rmsnorm_backward_kernel<<<BT, block_size, shared_bytes, stream>>>(
        dinp, dweight, scratch, dout, inp, weight, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

#endif // RMSNORM_CUH
