/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- For Q1.15: Uses scaled softmax cross-entropy with proper temperature scaling
- For Q1.31: Uses standard FP32 softmax, only input/output conversion to Q1.31
- The key insight: Q1.15 logits are stored normalized, but we need to scale them
  up before softmax to get proper probability distribution
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
// Q1.15 logit scaling: Q1.15 stores values in [-1, 1), but logits need [-24, 24] range
// This scale factor is applied before softmax to expand the effective range
// INCREASED from 16.0 to 24.0 for better softmax expressivity (lower entropy)
#define Q115_LOGIT_SCALE Q115_LOGITS_SCALE  // Use the scale from q115_common.cuh (24.0f)

// Temperature for softmax - lower = more confident predictions
// For Q1.15, we use temperature implicitly via Q115_LOGIT_SCALE
// A scale of 24 means: Q1.15 value of 0.5 -> actual logit of 12.0
#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
// Q1.31 has much higher precision, so we can use a larger logit range
// The scale factor converts Q1.31 [-1, 1) to actual logit range
#define Q131_LOGIT_SCALE 32.0f  // Q1.31 value of 0.5 -> actual logit of 16.0
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

#ifdef ENABLE_Q115
// Q1.15 Scaled Softmax Cross-Entropy Kernel
// Key insight: Q1.15 logits are stored normalized in [-1, 1), but we scale them
// up by Q115_LOGIT_SCALE before computing softmax for proper probability distribution
// Gradients are computed in float and then stored back with scaling for Q1.15 storage
template <bool WriteDLogits = true>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    q115_scaled_softmax_ce_kernel(floatX* logits, float* losses,
                                   const float dloss, const int* targets,
                                   int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    // Process in reverse order for cache hits on matmul data
    int64_t idx = gridDim.x - (blockIdx.x + 1);
    int target_ix = targets[idx];
    
    const floatX* logits_row = logits + idx * P;
    
    // Shared memory for max reduction (for numerical stability)
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Step 1: Find max logit (for numerical stability in softmax)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // Scale Q1.15 logits to actual logit range
        float logit = (float)logits_row[i] * Q115_LOGIT_SCALE;
        thread_max = fmaxf(thread_max, logit);
    }
    
    // Warp-level max reduction
    thread_max = warpReduceMax(thread_max);
    
    // Block-level reduction
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMax((int*)&shared_max, __float_as_int(thread_max));
    }
    __syncthreads();
    
    // First thread initializes shared_max properly
    if (threadIdx.x == 0) {
        // Handle atomicMax for floats properly
        float max_val = -INFINITY;
        for (int w = 0; w < blockDim.x / WARP_SIZE; w++) {
            // This is a simplification - real implementation would use proper float atomics
        }
    }
    
    // Use block reduction for max
    float block_max = blockReduce<warpReduceMax>(thread_max, false, -INFINITY);
    
    if (threadIdx.x == 0) {
        shared_max = block_max;
    }
    __syncthreads();
    float max_logit = shared_max;
    
    // Step 2: Compute sum of exp(logit - max)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float logit = (float)logits_row[i] * Q115_LOGIT_SCALE;
        thread_sum += expf(logit - max_logit);
    }
    
    // Block-level sum reduction
    float block_sum = blockReduce<warpReduceSum>(thread_sum);
    
    if (threadIdx.x == 0) {
        shared_sum = block_sum;
    }
    __syncthreads();
    float sum_exp = shared_sum;
    
    // Step 3: Compute loss (negative log probability of correct class)
    if (threadIdx.x == 0) {
        float target_logit = (float)logits_row[target_ix] * Q115_LOGIT_SCALE;
        float log_prob = target_logit - max_logit - logf(sum_exp);
        losses[idx] = -log_prob;  // Cross-entropy loss
    }
    __syncthreads();
    
    // Step 4: Compute and write gradients
    if (WriteDLogits) {
        float inv_sum = 1.0f / sum_exp;
        
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float logit = (float)logits_row[i] * Q115_LOGIT_SCALE;
            float prob = expf(logit - max_logit) * inv_sum;
            
            // Gradient: (prob - indicator) * dloss
            // For correct class, indicator = 1; for others, indicator = 0
            float indicator = (i == target_ix) ? 1.0f : 0.0f;
            float grad = (prob - indicator) * dloss;
            
            // Scale gradient back to Q1.15 range for storage
            // The gradient scale is the inverse of the logit scale
            // This ensures proper gradient flow through the network
            float grad_scaled = grad / Q115_LOGIT_SCALE;
            
            // Store gradient back to logits buffer
            logits[idx * P + i] = float_to_q115_scaled(grad_scaled, 1.0f);
        }
        
        // Zero out padding
        for (int i = V + threadIdx.x; i < P; i += blockDim.x) {
            logits[idx * P + i] = 0;
        }
    }
}
#endif  // ENABLE_Q115

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

#if defined(ENABLE_Q131) && !defined(FIXED_POINT_Q31)
    // For true Q1.31-logits mode: scale logits to expand dynamic range
    const float logit_scale = Q131_LOGIT_SCALE;
#elif defined(ENABLE_Q115)
    // For Q1.15 mode: scale logits to expand dynamic range
    // This is critical to break the ~7.x loss wall
    const float logit_scale = Q115_LOGIT_SCALE;
#else
    const float logit_scale = 1.0f;
#endif

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k] * logit_scale;
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k] * logit_scale;
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

#if defined(ENABLE_Q131) && !defined(FIXED_POINT_Q31)
    // For true Q1.31-logits mode: scale logits to expand dynamic range
    const float logit_scale = Q131_LOGIT_SCALE;
#elif defined(ENABLE_Q115)
    // For Q1.15 mode: scale logits to expand dynamic range
    const float logit_scale = Q115_LOGIT_SCALE;
#else
    const float logit_scale = 1.0f;
#endif

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] * logit_scale - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    // since the "logits" are overwritten to be in the [-1, 1] range and sp.Offset is sometimes smaller than -90
    // we errouneously end up computing exp^(90+) which gives us infinities in the loss! this is the fix.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] * logit_scale - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] * logit_scale - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    
#if defined(ENABLE_Q131)
    // For Q1.31: use standard classifier with Q1.31 logit scaling
    // Softmax/cross-entropy computed in FP32, only I/O uses Q1.31
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (floatX*)NULL, dloss, targets, B, T, V, P, write_dlogits);
#elif defined(ENABLE_Q115)
    // Use scaled softmax cross-entropy for Q1.15 mode
    // This properly handles the limited range of Q1.15 by scaling logits before softmax
    q115_scaled_softmax_ce_kernel<<<grid_size, block_size, 0, stream>>>(logits, losses, dloss, targets, B, T, V, P, write_dlogits);
#else
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (floatX*)NULL, dloss, targets, B, T, V, P, write_dlogits);
#endif
    cudaCheck(cudaGetLastError());
}
