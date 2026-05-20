/*
SFNet layers — SF16-aware, CUDA-optimized building blocks for the strict
Q1.15 forward / BF16 backward training regime.

This header introduces fused kernels designed so that every value written to
floatX storage in the forward pass already satisfies the Q1.15 boundary
[-0.999969..., +0.999969...] without relying on a separate clamp epilogue.

Provided kernels:
  - qhead_rmsnorm_forward / qhead_rmsnorm_backward
      Per-head RMSNorm over HD for Q (or K) with a learnable HD-vector weight.
      The packed QKV buffer is laid out per token as
        [ Q : NH*HD | K : NKV*HD | V : NKV*HD ]
      and we apply this kernel to the Q-slice and K-slice in turn (V is
      left untouched).  After this, RoPE can be applied; the combination is
      QK-Norm, which guarantees ‖q‖₂≈√HD and therefore
      qᵀk/√HD ∈ ~[-1, 1].  Softmax in attention is then perfectly
      Q1.15-stable.

  - rope_qk_forward / rope_qk_backward
      Pair-wise RoPE rotation applied to Q and K slices in the same packed
      QKV buffer.  A rotation preserves ‖·‖₂ exactly, so Q1.15 bounds are
      preserved.

  - tanh_glu_forward / tanh_glu_backward
      Bounded gated MLP non-linearity: out = tanh(gate_in) * up_in.
      tanh ∈ (-1, 1), up_in ∈ [-1, 1) ⇒ product ∈ (-1, 1).

  - scaled_residual_forward (+ optional fused next-block RMSNorm) and
    scaled_residual_backward
      Norm-preserving residual:  x' = c1 * x + c2 * branch,
      with c1 = sqrt(1 - α²), c2 = α.  If ‖x‖,‖branch‖ ≤ 1 then ‖x'‖ ≤ 1
      for any α ∈ [0, 1].  This eliminates residual-stream drift, which
      otherwise saturates SF16 networks as depth grows.

All forward kernels use streaming __ldcs / __stcs, warp-level block
reductions from cuda_utils.cuh, and SF16 quantization through the same
simulate_q115 / simulate_q131 helpers used elsewhere in the project.
Backward kernels use BF16/FP precision without SF16 quantization (matches
existing project convention; see quantize_sf_backward in matmul.cuh).
*/

#ifndef SFNET_LAYERS_CUH
#define SFNET_LAYERS_CUH

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

#if defined(ENABLE_Q115)
#include "q115_common.cuh"
#if defined(SF16_TRUE_FORWARD)
#include "q131_common.cuh"
#endif
#endif

// ---------------------------------------------------------------------------
// SF16 forward quantizer (forward only; backward stays BF16/FP).
// Sparsify + soft-limit before Q1.15 to reduce saturation overflow:
//   • deadband: |x| < 1/32 → exact zero (clears quant noise floor)
//   • soft knee: compress |x| > 0.75 inward before hard Q1.15 clip
// ---------------------------------------------------------------------------
#ifndef SFNET_ACT_DEADBAND
#define SFNET_ACT_DEADBAND (1.0f / 32.0f)
#endif
#ifndef SFNET_ACT_SOFT_LIMIT
#define SFNET_ACT_SOFT_LIMIT 0.75f
#endif

__device__ __forceinline__ float sfnet_q_fwd(float x) {
#if defined(ENABLE_Q115)
    if (fabsf(x) < SFNET_ACT_DEADBAND) {
        return 0.0f;
    }
    const float lim = SFNET_ACT_SOFT_LIMIT;
    if (x > lim) {
        x = lim + (x - lim) * 0.15f;
    } else if (x < -lim) {
        x = -lim + (x + lim) * 0.15f;
    }
    return simulate_q115(x);
#else
    return x;
#endif
}

__device__ __forceinline__ float sfnet_q_bwd(float x) { return x; }

// ===========================================================================
// 1) Per-head RMSNorm over HD (applied to the Q or K slice of packed QKV)
// ---------------------------------------------------------------------------
// Inputs:
//   qkv:     (BT, total_qkv) total_qkv = (NH + 2*NKV)*HD
//   norm_w:  (NHEAD_NORM, HD)            per-head HD-vector
//   slice_offset:  offset (in floatX) into each row at which the Q (or K)
//                  slice starts (0 for Q, NH*HD for K)
//   nhead_norm:    NH for Q, NKV for K
// One block per (b*t, head); HD ≤ 256 threads per block.
// Writes rstd of shape (BT, nhead_norm) for the backward pass.
// ===========================================================================

__global__ void qhead_rmsnorm_forward_kernel(
    floatX *__restrict__ qkv,
    float *__restrict__ rstd,                  // (BT, nhead_norm)
    const floatX *__restrict__ norm_w,         // (nhead_norm, HD)
    int total_qkv, int slice_offset,
    int nhead_norm, int HD, float eps)
{
    int bt = blockIdx.x;
    int h = blockIdx.y;
    floatX *row = qkv + (size_t)bt * total_qkv + slice_offset + (size_t)h * HD;
    const floatX *w = norm_w + (size_t)h * HD;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float xi = (float)__ldcs(&row[i]);
        ss += xi * xi;
    }
    float blk = blockReduce<warpReduceSum>(ss);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(blk / (float)HD + eps);
        if (rstd) rstd[(size_t)bt * nhead_norm + h] = s_rstd;
    }
    __syncthreads();
    float r = s_rstd;

    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float xi = (float)__ldcs(&row[i]);
        float wi = (float)__ldcs(&w[i]);
        float v = xi * r * wi;
        __stcs(&row[i], (floatX)sfnet_q_fwd(v));
    }
}

static inline void qhead_rmsnorm_forward(
    floatX *qkv, float *rstd, const floatX *norm_w,
    int BT, int total_qkv, int slice_offset,
    int nhead_norm, int HD, float eps, cudaStream_t stream)
{
    int threads = HD;
    if (threads > 256) threads = 256;
    threads = CEIL_DIV(threads, 32) * 32;
    dim3 grid(BT, nhead_norm);
    qhead_rmsnorm_forward_kernel<<<grid, threads, 0, stream>>>(
        qkv, rstd, norm_w, total_qkv, slice_offset, nhead_norm, HD, eps);
    cudaCheck(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm backward (BF16/FP, no SF16 quantization).
// dx_i      = r * w_i * dy_i  -  r^3 * x_i * dot / HD
// dw_i      += sum over (b*t)  of dy_i * x_i * r
// where dot = sum_i dy_i * w_i * x_i (computed per row).
//
// dqkv is in/out (holds dy on entry, dx on exit) at the slice.
// inp_qkv holds the pre-RMSNorm x (saved by caller).
// Per-row dot is reduced by a single block per (bt, head); dweight reduction
// is done in a small follow-up kernel because atomicAdd is not available
// for __nv_bfloat16.
// ---------------------------------------------------------------------------
__global__ void qhead_rmsnorm_backward_dinp_kernel(
    floatX *__restrict__ dqkv,                 // in: dy, out: dx
    const floatX *__restrict__ inp_qkv,        // x (pre-RMSNorm)
    const floatX *__restrict__ norm_w,
    const float *__restrict__ rstd,
    int total_qkv, int slice_offset,
    int nhead_norm, int HD)
{
    int bt = blockIdx.x;
    int h = blockIdx.y;
    floatX *drow = dqkv + (size_t)bt * total_qkv + slice_offset + (size_t)h * HD;
    const floatX *xrow = inp_qkv + (size_t)bt * total_qkv + slice_offset + (size_t)h * HD;
    const floatX *w = norm_w + (size_t)h * HD;
    float r = rstd[(size_t)bt * nhead_norm + h];

    // dot = sum_i dy_i * w_i * x_i
    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float dyi = (float)__ldcs(&drow[i]);
        float wi  = (float)__ldcs(&w[i]);
        float xi  = (float)__ldcs(&xrow[i]);
        thread_dot += dyi * wi * xi;
    }
    float blk = blockReduce<warpReduceSum>(thread_dot);
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = blk;
    __syncthreads();
    float dot_val = s_dot;

    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float dyi = (float)__ldcs(&drow[i]);
        float wi  = (float)__ldcs(&w[i]);
        float xi  = (float)__ldcs(&xrow[i]);
        float dxi = r * wi * dyi - r * r * r * xi * dot_val / (float)HD;
        __stcs(&drow[i], (floatX)sfnet_q_bwd(dxi));
    }
}

// Note: this dweight kernel must run BEFORE qhead_rmsnorm_backward_dinp_kernel,
// because once dinp overwrites dqkv we lose the dy values.
__global__ void qhead_rmsnorm_backward_dweight_kernel(
    floatX *__restrict__ dweight,              // (nhead_norm, HD) +=
    const floatX *__restrict__ dy_qkv,         // dqkv currently holding dy
    const floatX *__restrict__ inp_qkv,        // x (pre-RMSNorm)
    const float *__restrict__ rstd,            // (BT, nhead_norm)
    int BT, int total_qkv, int slice_offset,
    int nhead_norm, int HD)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nhead_norm * HD;
    if (idx >= total) return;
    int h = idx / HD;
    int i = idx % HD;

    float acc = 0.0f;
    for (int bt = 0; bt < BT; bt++) {
        size_t off = (size_t)bt * total_qkv + slice_offset + (size_t)h * HD + i;
        float dyi = (float)dy_qkv[off];
        float xi  = (float)inp_qkv[off];
        float r   = rstd[(size_t)bt * nhead_norm + h];
        acc += dyi * xi * r;
    }
    floatX *dw_ptr = dweight + (size_t)h * HD + i;
    float prev = (float)(*dw_ptr);
    *dw_ptr = (floatX)(prev + acc);
}

static inline void qhead_rmsnorm_backward(
    floatX *dqkv, floatX *dnorm_w,
    const floatX *inp_qkv, const floatX *norm_w, const float *rstd,
    int BT, int total_qkv, int slice_offset,
    int nhead_norm, int HD, cudaStream_t stream)
{
    // dweight first (reads dqkv as dy)
    {
        int total = nhead_norm * HD;
        int block = 256;
        int grid = CEIL_DIV(total, block);
        if (grid == 0) grid = 1;
        qhead_rmsnorm_backward_dweight_kernel<<<grid, block, 0, stream>>>(
            dnorm_w, dqkv, inp_qkv, rstd,
            BT, total_qkv, slice_offset, nhead_norm, HD);
        cudaCheck(cudaGetLastError());
    }
    // dinp (overwrites dqkv with dx)
    {
        int threads = HD;
        if (threads > 256) threads = 256;
        threads = CEIL_DIV(threads, 32) * 32;
        dim3 grid(BT, nhead_norm);
        qhead_rmsnorm_backward_dinp_kernel<<<grid, threads, 0, stream>>>(
            dqkv, inp_qkv, norm_w, rstd,
            total_qkv, slice_offset, nhead_norm, HD);
        cudaCheck(cudaGetLastError());
    }
}

// ===========================================================================
// 2) RoPE forward/backward applied to packed Q,K slices
// ---------------------------------------------------------------------------
// Inputs / outputs follow the same packed QKV layout as the rest of SFNet.
// ===========================================================================

__global__ void sfnet_rope_forward_kernel(
    floatX *__restrict__ qkv,
    const float2 *__restrict__ freqs,
    int B, int T, int NH, int NKV, int HD)
{
    int bt = blockIdx.x;
    int t = bt % T;
    int total_qkv = (NH + 2 * NKV) * HD;
    floatX *row = qkv + (size_t)bt * total_qkv;
    const float2 *f = freqs + (size_t)t * (HD / 2);

    for (int hp = threadIdx.x; hp < NH * (HD / 2); hp += blockDim.x) {
        int h = hp / (HD / 2), p = hp % (HD / 2);
        int base = h * HD + p * 2;
        float x0 = (float)row[base], x1 = (float)row[base + 1];
        float cv = f[p].x, sv = f[p].y;
        float y0 = x0 * cv - x1 * sv;
        float y1 = x0 * sv + x1 * cv;
        row[base]     = (floatX)sfnet_q_fwd(y0);
        row[base + 1] = (floatX)sfnet_q_fwd(y1);
    }
    for (int hp = threadIdx.x; hp < NKV * (HD / 2); hp += blockDim.x) {
        int h = hp / (HD / 2), p = hp % (HD / 2);
        int base = NH * HD + h * HD + p * 2;
        float x0 = (float)row[base], x1 = (float)row[base + 1];
        float cv = f[p].x, sv = f[p].y;
        float y0 = x0 * cv - x1 * sv;
        float y1 = x0 * sv + x1 * cv;
        row[base]     = (floatX)sfnet_q_fwd(y0);
        row[base + 1] = (floatX)sfnet_q_fwd(y1);
    }
}

__global__ void sfnet_rope_backward_kernel(
    floatX *__restrict__ dqkv,
    const float2 *__restrict__ freqs,
    int B, int T, int NH, int NKV, int HD)
{
    int bt = blockIdx.x;
    int t = bt % T;
    int total_qkv = (NH + 2 * NKV) * HD;
    floatX *row = dqkv + (size_t)bt * total_qkv;
    const float2 *f = freqs + (size_t)t * (HD / 2);

    for (int hp = threadIdx.x; hp < NH * (HD / 2); hp += blockDim.x) {
        int h = hp / (HD / 2), p = hp % (HD / 2);
        int base = h * HD + p * 2;
        float dy0 = (float)row[base], dy1 = (float)row[base + 1];
        float cv = f[p].x, sv = f[p].y;
        float dx0 =  dy0 * cv + dy1 * sv;
        float dx1 = -dy0 * sv + dy1 * cv;
        row[base]     = (floatX)dx0;
        row[base + 1] = (floatX)dx1;
    }
    for (int hp = threadIdx.x; hp < NKV * (HD / 2); hp += blockDim.x) {
        int h = hp / (HD / 2), p = hp % (HD / 2);
        int base = NH * HD + h * HD + p * 2;
        float dy0 = (float)row[base], dy1 = (float)row[base + 1];
        float cv = f[p].x, sv = f[p].y;
        float dx0 =  dy0 * cv + dy1 * sv;
        float dx1 = -dy0 * sv + dy1 * cv;
        row[base]     = (floatX)dx0;
        row[base + 1] = (floatX)dx1;
    }
}

static inline void rope_qk_forward(
    floatX *qkv, const float2 *freqs,
    int B, int T, int NH, int NKV, int HD, cudaStream_t stream)
{
    int threads = max(NH, NKV) * (HD / 2);
    if (threads > 256) threads = 256;
    threads = CEIL_DIV(threads, 32) * 32;
    sfnet_rope_forward_kernel<<<B * T, threads, 0, stream>>>(
        qkv, freqs, B, T, NH, NKV, HD);
    cudaCheck(cudaGetLastError());
}

static inline void rope_qk_backward(
    floatX *dqkv, const float2 *freqs,
    int B, int T, int NH, int NKV, int HD, cudaStream_t stream)
{
    int threads = max(NH, NKV) * (HD / 2);
    if (threads > 256) threads = 256;
    threads = CEIL_DIV(threads, 32) * 32;
    sfnet_rope_backward_kernel<<<B * T, threads, 0, stream>>>(
        dqkv, freqs, B, T, NH, NKV, HD);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// 3) Tanh-GLU MLP non-linearity:  out = tanh(gate_in) * up_in
// ===========================================================================
__global__ void tanh_glu_forward_kernel(
    floatX *__restrict__ out,
    const floatX *__restrict__ gate_in,
    const floatX *__restrict__ up_in,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float g = (float)__ldcs(&gate_in[i]);
        float u = (float)__ldcs(&up_in[i]);
        float tg = tanhf(g);
        float v = tg * u;
        __stcs(&out[i], (floatX)sfnet_q_fwd(v));
    }
}

static inline void tanh_glu_forward(
    floatX *out, const floatX *gate_in, const floatX *up_in, int N,
    cudaStream_t stream)
{
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    tanh_glu_forward_kernel<<<grid, block, 0, stream>>>(out, gate_in, up_in, N);
    cudaCheck(cudaGetLastError());
}

__global__ void tanh_glu_backward_kernel(
    floatX *__restrict__ d_gate,
    floatX *__restrict__ d_up,
    const floatX *__restrict__ d_out,
    const floatX *__restrict__ gate_in,
    const floatX *__restrict__ up_in,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float dy = (float)__ldcs(&d_out[i]);
        float g  = (float)__ldcs(&gate_in[i]);
        float u  = (float)__ldcs(&up_in[i]);
        float tg = tanhf(g);
        float dtanh = 1.0f - tg * tg;
        __stcs(&d_gate[i], (floatX)(dy * u * dtanh));
        __stcs(&d_up[i],   (floatX)(dy * tg));
    }
}

static inline void tanh_glu_backward(
    floatX *d_gate, floatX *d_up,
    const floatX *d_out, const floatX *gate_in, const floatX *up_in,
    int N, cudaStream_t stream)
{
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    tanh_glu_backward_kernel<<<grid, block, 0, stream>>>(
        d_gate, d_up, d_out, gate_in, up_in, N);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// 4) Norm-preserving (scaled) residual
//   x' = c1 * x + c2 * branch    with   c1 = sqrt(1 - alpha^2), c2 = alpha
// ===========================================================================
__global__ void scaled_residual_forward_kernel(
    floatX *__restrict__ out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ branch,
    float c1, float c2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float xi = (float)__ldcs(&x[i]);
        float bi = (float)__ldcs(&branch[i]);
        float v = c1 * xi + c2 * bi;
        __stcs(&out[i], (floatX)sfnet_q_fwd(v));
    }
}

static inline void scaled_residual_forward(
    floatX *out, const floatX *x, const floatX *branch,
    float alpha, int N, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    scaled_residual_forward_kernel<<<grid, block, 0, stream>>>(
        out, x, branch, c1, c2, N);
    cudaCheck(cudaGetLastError());
}

// Fused: x_out = c1*x + c2*branch ; then per-row RMSNorm(x_out)*w into `normed`.
// One block per row (B*T).
__global__ void fused_scaled_residual_rmsnorm_kernel(
    floatX *__restrict__ x_out,
    floatX *__restrict__ normed,
    float *__restrict__ rstd,
    const floatX *__restrict__ x,
    const floatX *__restrict__ branch,
    const floatX *__restrict__ rmsw,
    float c1, float c2, int C, float eps)
{
    int bt = blockIdx.x;
    const floatX *xrow = x      + (size_t)bt * C;
    const floatX *brow = branch + (size_t)bt * C;
    floatX *orow = x_out  + (size_t)bt * C;
    floatX *nrow = normed + (size_t)bt * C;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = c1 * (float)__ldcs(&xrow[i]) + c2 * (float)__ldcs(&brow[i]);
        v = sfnet_q_fwd(v);
        __stcs(&orow[i], (floatX)v);
        ss += v * v;
    }
    float blk = blockReduce<warpReduceSum>(ss);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(blk / (float)C + eps);
        if (rstd) rstd[bt] = s_rstd;
    }
    __syncthreads();
    float r = s_rstd;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = (float)__ldcs(&orow[i]);
        float w = (float)__ldcs(&rmsw[i]);
        float y = v * r * w;
        __stcs(&nrow[i], (floatX)sfnet_q_fwd(y));
    }
}

static inline void fused_scaled_residual_rmsnorm(
    floatX *x_out, floatX *normed, float *rstd,
    const floatX *x, const floatX *branch, const floatX *rmsw,
    float alpha, int BT, int C, float eps, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int threads = C;
    if (threads > 512) threads = 512;
    threads = CEIL_DIV(threads, 32) * 32;
    fused_scaled_residual_rmsnorm_kernel<<<BT, threads, 0, stream>>>(
        x_out, normed, rstd, x, branch, rmsw, c1, c2, C, eps);
    cudaCheck(cudaGetLastError());
}

// 3-input fused scaled residual + (optional) next-block RMSNorm.
//   x_out = c1 * x + c2 * (a + b)
// In SFNet a = attn_out and b = mlp_out (parallel-block branches).
// We never materialise (a+b) on its own; instead we fold the addition into
// the residual-quantize epilogue so intermediate values never need to fit
// inside the Q1.15 storage range.
__global__ void scaled_residual_3way_forward_kernel(
    floatX *__restrict__ x_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ a,
    const floatX *__restrict__ b,
    float c1, float c2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float xi = (float)__ldcs(&x[i]);
        float ai = (float)__ldcs(&a[i]);
        float bi = (float)__ldcs(&b[i]);
        float v = c1 * xi + c2 * (ai + bi);
        __stcs(&x_out[i], (floatX)sfnet_q_fwd(v));
    }
}

static inline void scaled_residual_3way_forward(
    floatX *x_out, const floatX *x, const floatX *a, const floatX *b,
    float alpha, int N, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    scaled_residual_3way_forward_kernel<<<grid, block, 0, stream>>>(
        x_out, x, a, b, c1, c2, N);
    cudaCheck(cudaGetLastError());
}

// Fused 3-way scaled residual + next-block RMSNorm in a single memory pass.
//   x_out  = c1*x + c2*(a + b)
//   normed = (x_out / rms(x_out)) * w
__global__ void fused_scaled_residual_3way_rmsnorm_kernel(
    floatX *__restrict__ x_out,
    floatX *__restrict__ normed,
    float *__restrict__ rstd,
    const floatX *__restrict__ x,
    const floatX *__restrict__ a,
    const floatX *__restrict__ b,
    const floatX *__restrict__ rmsw,
    float c1, float c2, int C, float eps)
{
    int bt = blockIdx.x;
    const floatX *xrow = x + (size_t)bt * C;
    const floatX *arow = a + (size_t)bt * C;
    const floatX *brow = b + (size_t)bt * C;
    floatX *orow = x_out + (size_t)bt * C;
    floatX *nrow = normed + (size_t)bt * C;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = c1 * (float)__ldcs(&xrow[i])
                + c2 * ((float)__ldcs(&arow[i]) + (float)__ldcs(&brow[i]));
        v = sfnet_q_fwd(v);
        __stcs(&orow[i], (floatX)v);
        ss += v * v;
    }
    float blk = blockReduce<warpReduceSum>(ss);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(blk / (float)C + eps);
        if (rstd) rstd[bt] = s_rstd;
    }
    __syncthreads();
    float r = s_rstd;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = (float)__ldcs(&orow[i]);
        float w = (float)__ldcs(&rmsw[i]);
        float y = v * r * w;
        __stcs(&nrow[i], (floatX)sfnet_q_fwd(y));
    }
}

static inline void fused_scaled_residual_3way_rmsnorm(
    floatX *x_out, floatX *normed, float *rstd,
    const floatX *x, const floatX *a, const floatX *b, const floatX *rmsw,
    float alpha, int BT, int C, float eps, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int threads = C;
    if (threads > 512) threads = 512;
    threads = CEIL_DIV(threads, 32) * 32;
    fused_scaled_residual_3way_rmsnorm_kernel<<<BT, threads, 0, stream>>>(
        x_out, normed, rstd, x, a, b, rmsw, c1, c2, C, eps);
    cudaCheck(cudaGetLastError());
}

// Backward for 3-way scaled residual x' = c1*x + c2*(a+b)
//   dx += c1 * d_out
//   da := c2 * d_out
//   db := c2 * d_out
__global__ void scaled_residual_3way_backward_kernel(
    floatX *__restrict__ dx,
    floatX *__restrict__ da,
    floatX *__restrict__ db,
    const floatX *__restrict__ d_out,
    float c1, float c2, int N, bool accumulate_dx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float gy = (float)__ldcs(&d_out[i]);
        float gb = c2 * gy;
        __stcs(&da[i], (floatX)gb);
        __stcs(&db[i], (floatX)gb);
        if (accumulate_dx) {
            float prev = (float)__ldcs(&dx[i]);
            __stcs(&dx[i], (floatX)(prev + c1 * gy));
        } else {
            __stcs(&dx[i], (floatX)(c1 * gy));
        }
    }
}

static inline void scaled_residual_3way_backward(
    floatX *dx, floatX *da, floatX *db, const floatX *d_out,
    float alpha, int N, bool accumulate_dx, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    scaled_residual_3way_backward_kernel<<<grid, block, 0, stream>>>(
        dx, da, db, d_out, c1, c2, N, accumulate_dx);
    cudaCheck(cudaGetLastError());
}

// Backward for scaled residual x' = c1*x + c2*branch
//   dx       += c1 * d_out
//   dbranch  := c2 * d_out
__global__ void scaled_residual_backward_kernel(
    floatX *__restrict__ dx,
    floatX *__restrict__ dbranch,
    const floatX *__restrict__ d_out,
    float c1, float c2, int N, bool accumulate_dx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float gy = (float)__ldcs(&d_out[i]);
        __stcs(&dbranch[i], (floatX)(c2 * gy));
        if (accumulate_dx) {
            float prev = (float)__ldcs(&dx[i]);
            __stcs(&dx[i], (floatX)(prev + c1 * gy));
        } else {
            __stcs(&dx[i], (floatX)(c1 * gy));
        }
    }
}

static inline void scaled_residual_backward(
    floatX *dx, floatX *dbranch, const floatX *d_out,
    float alpha, int N, bool accumulate_dx, cudaStream_t stream)
{
    float c1 = sqrtf(fmaxf(0.0f, 1.0f - alpha * alpha));
    float c2 = alpha;
    int block = 256;
    int grid = CEIL_DIV(N, block);
    if (grid > 65535) grid = 65535;
    if (grid == 0) grid = 1;
    scaled_residual_backward_kernel<<<grid, block, 0, stream>>>(
        dx, dbranch, d_out, c1, c2, N, accumulate_dx);
    cudaCheck(cudaGetLastError());
}

#endif // SFNET_LAYERS_CUH
