/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- For Q1.15: Uses scaled softmax cross-entropy — logits stored as BF16 are
  multiplied by Q115_LOGIT_SCALE before softmax to expand the effective range.
- For Q1.31: Uses standard FP32 softmax, only input/output conversion to Q1.31
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
// Q1.15 logit scaling: logits are stored as BF16 in a simulated [-1, 1) range.
// Multiply by this scale before softmax to expand effective logit range to ~[-24, 24].
#if defined(SF16_TRUE_FORWARD)
#define Q115_LOGIT_SCALE 1.0f
#else
#define Q115_LOGIT_SCALE Q115_LOGITS_SCALE  // 24.0f from q115_common.cuh
#endif

__device__ __forceinline__ float q115_logit_scale_for_vocab(int V) {
#if defined(SF16_TRUE_FORWARD)
  // In strict true-forward mode, large-vocab softmaxes need expansion.
  return (V >= 100000) ? Q115_LOGITS_SCALE : Q115_LOGIT_SCALE;
#else
  (void)V;
  return Q115_LOGIT_SCALE;
#endif
}

#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
// Q1.31 has much higher precision; scale converts [-1,1) to actual logit range.
#define Q131_LOGIT_SCALE 32.0f
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

struct SoftmaxParams {
  float Scale;
  float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx,
                                                    const floatX *inp, int V,
                                                    int P) {
  // one row of inp, i.e. inp[idx, :] of shape (V,)
  const floatX *x = inp + idx * P;
  float thread_maxval = -INFINITY;
  float thread_sumval = 0.0f;
  int i = (V + x128::size - 1) / x128::size + threadIdx.x - blockDim.x;

#if defined(ENABLE_Q131) && !defined(FIXED_POINT_Q31)
  const float logit_scale = Q131_LOGIT_SCALE;
#elif defined(ENABLE_Q115)
  const float logit_scale = q115_logit_scale_for_vocab(V);
#else
  const float logit_scale = 1.0f;
#endif

  // special-case loop: unaligned tail elements at end of row
  while ((i + 1) * x128::size > V) {
    for (int k = 0; k < x128::size; ++k) {
      if (i * x128::size + k >= V) { break; }
      float v = (float)x[i * x128::size + k] * logit_scale;
      float old_maxval = thread_maxval;
      thread_maxval = fmaxf(thread_maxval, v);
      thread_sumval *= expf((old_maxval - thread_maxval));
      thread_sumval += expf(v - thread_maxval);
    }
    i -= blockDim.x;
  }

  // main loop (no bounds check needed)
  for (; i >= 0; i -= blockDim.x) {
    x128 packed_x = load128(x + i * x128::size);
    for (int k = 0; k < x128::size; ++k) {
      float v = (float)packed_x[k] * logit_scale;
      float old_maxval = thread_maxval;
      thread_maxval = fmaxf(thread_maxval, v);
      thread_sumval *= expf((old_maxval - thread_maxval));
      thread_sumval += expf(v - thread_maxval);
    }
  }

  // Block Max -> adjust sumval -> Block Sum
  float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
  thread_sumval *= expf(thread_maxval - block_maxval);
  float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

  return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX *logits, float *losses, floatX *probs,
                             const float dloss, const int *targets, int B,
                             int T, int V, int P,
                             std::bool_constant<WriteDLogits>) {
  // idx in reverse order for cache hits on matmul data
  int64_t idx = gridDim.x - (blockIdx.x + 1);
  int ix = targets[idx];

#if defined(ENABLE_Q131) && !defined(FIXED_POINT_Q31)
  const float logit_scale = Q131_LOGIT_SCALE;
#elif defined(ENABLE_Q115)
  const float logit_scale = q115_logit_scale_for_vocab(V);
#else
  const float logit_scale = 1.0f;
#endif

  // softmax parameters (reads B * T * V logits)
  SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

  // compute loss from the target logit (single-threaded)
  if (threadIdx.x == 0) {
    float prob =
        expf((float)logits[idx * P + ix] * logit_scale - sp.Offset) * sp.Scale;
    losses[idx] -= logf(prob);
  }

  // Synchronise before overwriting logits with gradients.
  // Without this there is a race: the logits read above to compute loss are
  // concurrently overwritten by gradient writes below, producing wrong loss.
  __syncthreads();

  // write logit gradients (overwrites logits buffer in-place)
  const floatX *logits_vec = logits + idx * P;
  for (int i = threadIdx.x; i < V / x128::size; i += blockDim.x) {
    x128 packed_logits_vec = load128(logits_vec + i * x128::size);
    x128 packed_probs;
    for (int k = 0; k < x128::size; ++k) {
      int element = i * x128::size + k;
      float prob = expf((float)packed_logits_vec[k] * logit_scale - sp.Offset) *
                   sp.Scale;
      packed_probs[k] = (floatX)prob;
      float indicator = (element == ix) ? 1.0f : 0.0f;
      // dL/d(logit_stored) = (prob - indicator) * dloss * logit_scale
      // (chain rule through logit_used = logit_stored * logit_scale)
      packed_logits_vec[k] = (floatX)((prob - indicator) * dloss * logit_scale);
    }
    if (WriteDLogits) {
      store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
    }
    if (WriteProbs) {
      store128(probs + idx * P + i * x128::size, packed_probs);
    }
  }

  // handle remaining elements after last multiple of x128::size
  int unaligned_start = V & ~(x128::size - 1);
  for (int i = threadIdx.x + unaligned_start; i < V; i++) {
    float prob =
        expf((float)logits_vec[i] * logit_scale - sp.Offset) * sp.Scale;
    float indicator = (i == ix) ? 1.0f : 0.0f;
    float dlogit = (prob - indicator) * dloss * logit_scale;
    if (WriteDLogits) {
      __stcs(logits + idx * P + i, (floatX)dlogit);
    }
    if (WriteProbs) {
      probs[idx * P + i] = (floatX)prob;
    }
  }
}

// ----------------------------------------------------------------------------
// kernel launcher

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
void fused_classifier(Type *logits, float *losses, const float dloss,
                      const int *targets, int B, int T, int V, int P,
                      std::bool_constant<WriteDLogits> write_dlogits,
                      cudaStream_t stream) {
  NVTX_RANGE_FN();
  const int block_size = 1024;
  const int N = B * T;
  const int grid_size = N;

  // For all precision modes (BF16, Q1.15, Q1.31) we use fused_classifier_kernel5.
  // The logit_scale defined per-mode inside the kernel handles the Q115/Q131
  // range expansion. The broken q115_scaled_softmax_ce_kernel has been removed.
  fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(
      logits, losses, (floatX *)NULL, dloss, targets, B, T, V, P,
      write_dlogits);
  cudaCheck(cudaGetLastError());
}
