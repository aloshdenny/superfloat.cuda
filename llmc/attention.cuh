/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#include <assert.h>
// llmc internal imports
#include "cublas_common.h"
#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
#if defined(SF16_TRUE_FORWARD)
#include "q131_common.cuh"
#endif
#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
#endif

constexpr int FLASH_WARPS_PER_BLOCK = 4;
constexpr int FLASH_K_TILE = 32;
constexpr int FLASH_MAX_HEAD_DIM = 128;

__device__ __forceinline__ float quantize_attention_score(float x) {
#if defined(ENABLE_Q131)
  return simulate_q131(x * 8.0f);
#elif defined(ENABLE_Q115)
  return simulate_q115(x);
#else
  return x;
#endif
}

__device__ __forceinline__ float quantize_attention_output(float x) {
#if defined(ENABLE_Q131)
  return simulate_q131(x);
#elif defined(ENABLE_Q115)
#if defined(SF16_TRUE_FORWARD)
  x = simulate_q131(x);
#endif
  return simulate_q115(x);
#else
  return x;
#endif
}

// ----------------------------------------------------------------------------
// CUDA kernels

// inputs floatX, outputs FP32 (for current FP32-only activation path for this
// WIP)
__global__ void permute_kernel(floatX *q, floatX *k, floatX *v,
                               const floatX *inp, int B, int N, int NH, int d) {
  // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
  // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;
  int inp_idx =
      (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
  q[idx] = __ldcs(&inp[inp_idx]);
  k[idx] = __ldcs(&inp[inp_idx + NH * d]);
  v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

__global__ void permute_kernel_backward(floatX *dinp, const floatX *dq,
                                        const floatX *dk, const floatX *dv,
                                        int B, int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;

  int inp_idx =
      (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
  dinp[inp_idx] = dq[idx];
  dinp[inp_idx + NH * d] = dk[idx];
  dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX *inp, floatX *out, int B, int N, int NH,
                                 int d) {
  // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;
  int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
  out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX *dinp, const floatX *dout,
                                          int B, int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;
  int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
  dinp[idx] = (floatX)dout[other_idx];
}

template <int WARPS_PER_BLOCK, int K_TILE, int MAX_HEAD_DIM>
__global__ void flash_attention_tiled_forward_kernel(
  floatX *out, floatX *qkvr, const floatX *inp, int B, int T, int C, int NH,
  int HS) {
  static_assert(MAX_HEAD_DIM % WARP_SIZE == 0,
                "MAX_HEAD_DIM must be a multiple of warp size");
  constexpr int D_VECS = MAX_HEAD_DIM / WARP_SIZE;

  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  int q_row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int bh = blockIdx.y;
  bool active_row = (q_row < T) && (bh < B * NH);

  int b = bh / NH;
  int h = bh % NH;
  floatX *q = qkvr + 0 * B * T * C;
  floatX *k = qkvr + 1 * B * T * C;
  floatX *v = qkvr + 2 * B * T * C;

  extern __shared__ char smem_raw[];
  floatX *k_s = reinterpret_cast<floatX *>(smem_raw);
  floatX *v_s = k_s + K_TILE * MAX_HEAD_DIM;
  float *p_s = reinterpret_cast<float *>(v_s + K_TILE * MAX_HEAD_DIM);
  float *warp_probs = p_s + warp_id * (K_TILE + 1);

  float q_reg[D_VECS];
  float o_reg[D_VECS];
  for (int i = 0; i < D_VECS; i++) {
    q_reg[i] = 0.0f;
    o_reg[i] = 0.0f;
  }

  const float flt_max =
      340282346638528859811704183484516925440.0f; // avoid float.h include
  float m = -flt_max;
  float l = 0.0f;
  float inv_sqrt_hs = rsqrtf((float)HS);

  if (active_row) {
    size_t q_base = ((size_t)bh * T + q_row) * HS;
    size_t inp_token_base = ((size_t)b * T + q_row) * 3 * C;
    for (int i = 0; i < D_VECS; i++) {
      int d = lane_id + i * WARP_SIZE;
      if (d < HS) {
        size_t h_offset = (size_t)h * HS + d;
        q_reg[i] = (float)__ldcs(inp + inp_token_base + h_offset);
        q[q_base + d] = (floatX)q_reg[i];
        k[q_base + d] = __ldcs(inp + inp_token_base + C + h_offset);
        v[q_base + d] = __ldcs(inp + inp_token_base + 2 * C + h_offset);
      }
    }
  }

  for (int k0 = 0; k0 < T; k0 += K_TILE) {
    int tile_count = min(K_TILE, T - k0);

    for (int idx = threadIdx.x; idx < tile_count * HS; idx += blockDim.x) {
      int t_j = idx / HS;
      int d = idx % HS;
      size_t token_base = ((size_t)b * T + (k0 + t_j)) * 3 * C;
      size_t h_offset = (size_t)h * HS + d;
      k_s[t_j * MAX_HEAD_DIM + d] = __ldcs(inp + token_base + C + h_offset);
      v_s[t_j * MAX_HEAD_DIM + d] =
          __ldcs(inp + token_base + 2 * C + h_offset);
    }
    __syncthreads();

    if (active_row && k0 <= q_row) {
      int valid = min(tile_count, q_row - k0 + 1);

      for (int t_j = 0; t_j < valid; t_j++) {
        float dot = 0.0f;
        for (int i = 0; i < D_VECS; i++) {
          int d = lane_id + i * WARP_SIZE;
          if (d < HS) {
            // k_s is shared memory; use a regular load, not global-cache
            // intrinsics.
            dot += q_reg[i] * (float)k_s[t_j * MAX_HEAD_DIM + d];
          }
        }
        dot = warpReduceSum(dot);
        if (lane_id == 0) {
          float score = quantize_attention_score(dot * inv_sqrt_hs);
          warp_probs[t_j] = score;
        }
      }
      __syncwarp();

      if (lane_id == 0) {
        float tile_max = -flt_max;
        for (int t_j = 0; t_j < valid; t_j++) {
          tile_max = fmaxf(tile_max, warp_probs[t_j]);
        }

        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);
        float l_new = l * alpha;

        for (int t_j = 0; t_j < valid; t_j++) {
          float p = expf(warp_probs[t_j] - m_new);
          warp_probs[t_j] = p;
          l_new += p;
        }

        float inv_l_new = 1.0f / fmaxf(l_new, 1e-20f);
        float old_scale = (l * alpha) * inv_l_new;
        for (int t_j = 0; t_j < valid; t_j++) {
          warp_probs[t_j] *= inv_l_new;
        }
        warp_probs[K_TILE] = old_scale;

        m = m_new;
        l = l_new;
      }
      __syncwarp();

      float old_scale = warp_probs[K_TILE];
      for (int i = 0; i < D_VECS; i++) {
        o_reg[i] *= old_scale;
      }

      for (int t_j = 0; t_j < valid; t_j++) {
        float p = warp_probs[t_j];
        for (int i = 0; i < D_VECS; i++) {
          int d = lane_id + i * WARP_SIZE;
          if (d < HS) {
            // v_s is shared memory; use a regular load, not global-cache
            // intrinsics.
            o_reg[i] += p * (float)v_s[t_j * MAX_HEAD_DIM + d];
          }
        }
      }
    }

    __syncthreads();
  }

  if (active_row) {
    size_t out_base = ((size_t)b * T + q_row) * C + h * HS;
    for (int i = 0; i < D_VECS; i++) {
      int d = lane_id + i * WARP_SIZE;
      if (d < HS) {
        float out_val = quantize_attention_output(o_reg[i]);
        __stcs(out + out_base + d, (floatX)out_val);
      }
    }
  }
}

__global__ void softmax_forward_kernel5(floatX *out, float inv_temperature,
                                        const floatX *inp, int N, int T) {
  // inp, out shape: (N, T, T), where N = B * NH
  // fuses the multiplication by scale inside attention
  // directly autoregressive, so we only compute the lower triangular part
  // uses the online softmax algorithm
  // NOTE: For Q1.15 mode, all softmax computation is done in FP32 to prevent
  // attention collapse
  assert(T % 4 == 0);
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  int num_warps = blockDim.x / WARP_SIZE;

  // micro-optimization: we iterate backwards so that
  // after the softmax backward operation completes, the cache retains the
  // part of the matrix close to the upper left corner, which benefits the
  // matmul operation that immediately follows.
  // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); //
  // forward order
  int idx =
      (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; // backward order
  if (idx >= N * T) {
    return;
  }
  int own_pos = idx % T;
  int pos_by_4 = own_pos / 4;

  // one row of inp, i.e. inp[idx, :] of shape (T,)
  const floatX *x = inp + idx * T;

  // not INF, so we don't get NaNs accidentally when subtracting two values.
  const float flt_max =
      340282346638528859811704183484516925440.0f; // to avoid including float.h
  float maxval = -flt_max;
  float sumval = 0.0f;

  const floatX *x_aligned =
      reinterpret_cast<const floatX *>(__builtin_assume_aligned(x, 16));
  for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
    float regarray[4];
    for (int k = 0; k < 4; ++k) {
      regarray[k] = quantize_attention_score((float)x_aligned[4 * i + k]);
    }
    float old_maxval = maxval;
    for (int k = 0; k < 4; ++k) {
      maxval = fmaxf(maxval, regarray[k]);
    }
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    for (int k = 0; k < 4; ++k) {
      sumval += expf(inv_temperature * (regarray[k] - maxval));
    }
  }

  if (4 * pos_by_4 + lane_id <= own_pos) {
    float old_maxval = maxval;
    float scaled_val = quantize_attention_score((float)x[4 * pos_by_4 + lane_id]);
    maxval = fmaxf(maxval, scaled_val);
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    sumval += expf(inv_temperature * (scaled_val - maxval));
  }

  float global_maxval = warpReduceMax(maxval);
  sumval *= expf(inv_temperature * (maxval - global_maxval));

  float sum = warpReduceSum(sumval);
  float norm = 1.f / sum;

  // divide the whole row by the sum
  for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
    // recalculation is faster than doing the round-trip through memory.
    float ev = expf(inv_temperature *
                    (quantize_attention_score((float)__ldcs(x + i)) -
                     global_maxval));
    __stcs(out + idx * T + i, (floatX)quantize_attention_output(ev * norm));
  }
}

__global__ void softmax_autoregressive_backward_inplace_kernel(
    floatX *datt, const floatX *att, int B, int T, int C, float scale) {
  constexpr const int BlockSize = 256;
  constexpr int T_per_block = 4;

  // go through blocks in reverse order, so the slowest block starts first
  int t0 = T - 1 - T_per_block * blockIdx.x;
  int idx = blockIdx.y;

  att += idx * T * T;
  datt += idx * T * T;

  for (int to = 0; to < T_per_block; ++to) {
    int t = t0 - to;
    if (t < 0)
      return;
    const floatX *att_bth = att + t * T;
    const floatX *datt_bth = datt + t * T;
    floatX *dpreatt_bth = datt + t * T;

    float local_sum = 0;
    for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
      local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
    }

    local_sum = blockReduce<warpReduceSum>(local_sum);

    for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
      // don't touch the cache. Some parts will still be here from the previous
      // loop, and we want to exploit those.
      if (t3 <= t) {
        float acc = (float)__ldcs(att_bth + t3) *
                    ((float)__ldcs(datt_bth + t3) - local_sum);
        __stcs(dpreatt_bth + t3, (floatX)(scale * acc));
      } else {
        // explicitly set non-causal elements to zero
        __stcs(dpreatt_bth + t3, (floatX)0.f);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(floatX *out, floatX *qkvr, floatX *att, floatX *inp,
                       int B, int T, int C, int NH, cudaStream_t stream) {
  NVTX_RANGE_FN();
  (void)att; // Forward path no longer materializes T x T attention matrix.
  // Note: `inp` is not needed for backward pass, so we re-use it as a scratch
  // buffer. Its contents will be overwritten by this function.
  const int block_size = 256;

  // inp is (B, T, 3C) QKV
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  const int HS = C / NH; // head size

  // FlashAttention-style forward pass: tiled QK^T + online softmax + V
  // accumulation in shared memory, no T x T materialization.
  // The kernel reads packed QKV projection output directly from `inp`, writes
  // qkvr (for backward), and writes final attention output.
  if (HS <= FLASH_MAX_HEAD_DIM && HS % 8 == 0) {
    dim3 grid(CEIL_DIV(T, FLASH_WARPS_PER_BLOCK), B * NH);
    dim3 block(FLASH_WARPS_PER_BLOCK * WARP_SIZE);
    int shmem_size =
        2 * FLASH_K_TILE * FLASH_MAX_HEAD_DIM * sizeof(floatX) +
        FLASH_WARPS_PER_BLOCK * (FLASH_K_TILE + 1) * sizeof(float);
    flash_attention_tiled_forward_kernel<FLASH_WARPS_PER_BLOCK, FLASH_K_TILE,
                                         FLASH_MAX_HEAD_DIM>
        <<<grid, block, shmem_size, stream>>>(out, qkvr, inp, B, T, C, NH, HS);
    cudaCheck(cudaGetLastError());
    return;
  }

  // Fallback path for uncommon head sizes.
  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  floatX *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  int total_threads = B * NH * T * HS;
  int num_blocks = CEIL_DIV(total_threads, block_size);
  permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH,
                                                        HS);

  floatX *preatt = inp; // reuse inp as scratch buffer
  matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH,
                  T * HS, T * HS, T * T);

  // multiply all elements of preatt elementwise by scale
  float scale = 1.f / sqrtf(HS);
  int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
  softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(
      att, scale, preatt, B * NH, T);

  // new approach: first cuBLAS another batched matmul
  floatX *vaccum = inp;
  // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
  matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false,
                  B * NH, T * HS, T * T, T * HS);

  // now unpermute
  // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head
  // outputs side by side
  num_blocks = CEIL_DIV(B * T * C, block_size);
  unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH,
                                                          HS);
  cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) ->
// vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX *dinp, floatX *dqkvr, floatX *datt,
                        floatX *scratch, const floatX *dout, const floatX *qkvr,
                        floatX *att, int B, int T, int C, int NH,
                        cudaStream_t stream) {
  NVTX_RANGE_FN();
  const int block_size = 256;
  const int HS = C / NH; // head size

  // unpack convenience pointers into q, k, v
  const floatX *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  floatX *dq, *dk, *dv;
  dq = dqkvr + 0 * B * T * C;
  dk = dqkvr + 1 * B * T * C;
  dv = dqkvr + 2 * B * T * C;

  // Recompute attention probabilities from Q,K for backward.
  // Forward uses a tiled FlashAttention-style kernel and avoids T x T
  // materialization, so we regenerate probabilities here when gradients are
  // needed.
  floatX *preatt = datt;
  matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH,
                  T * HS, T * HS, T * T);
  float scale_fwd = 1.f / sqrtf((float)HS);
  int softmax_grid = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
  softmax_forward_kernel5<<<softmax_grid, block_size, 0, stream>>>(
      att, scale_fwd, preatt, B * NH, T);

  // backward through the unpermute operation
  int num_blocks = CEIL_DIV(B * T * C, block_size);
  unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
      scratch, dout, B, T, NH, HS);
  // backward into datt
  matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false,
                  B * NH, T * HS, T * HS, T * T);
  // backward into dv
  matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true,
                  B * NH, T * HS, T * T, T * HS);

#if defined(ENABLE_Q131)
  const float att_scale = 8.0f;
#elif defined(ENABLE_Q115)
  const float att_scale = 1.0f;
#else
  const float att_scale = 1.0f;
#endif

  const float scale = (1.0f / sqrtf((float)HS)) * att_scale;
  // backward into preatt. this is an in-place operation; datt turns into
  // dpreatt here
  softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(
      datt, att, B, T, C, scale);
  const floatX *dpreatt = datt;
  // backward into q
  matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false,
                  B * NH, T * HS, T * T, T * HS);
  // backward into k
  matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true,
                  B * NH, T * HS, T * T, T * HS);
  // backward into inp
  num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
  permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
      dinp, dq, dk, dv, B, T, NH, HS);
  cudaCheck(cudaGetLastError());
}
