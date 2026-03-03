/*
Q1.15 Fixed-Point Arithmetic Utilities for CUDA
Provides device functions for Q1.15 format operations

IMPROVED VERSION with Block Floating Point (BFP) scaling:
- Q1.15 stores normalized values in [-1, 1)
- Each tensor has an associated float scale factor
- Effective value = Q1.15_value * scale_factor
- This allows much larger dynamic range while keeping 16-bit storage
*/

#ifndef Q115_COMMON_CUH
#define Q115_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Q1.15 type definition: signed 16-bit integer representing range [-1, 1)
typedef int16_t q115_t;

// Q1.15 constants
#define Q115_SCALE 32768.0f          // 2^15
#define Q115_MAX 32767               // Maximum Q1.15 value (0.999969...)
#define Q115_MIN -32768              // Minimum Q1.15 value (-1.0)
#define Q115_OVERFLOW_THRESHOLD 0.999f  // Clamp values to prevent overflow

// Q1.15 smallest non-zero value is approximately 3e-5 (1/32768)
// Using epsilon below this causes numerical instability
#define Q115_MIN_NONZERO (1.0f / Q115_SCALE)

// ============================================================================
// Dynamic Scaling Configuration for Block Floating Point (BFP)
// Each tensor type has its own scale factor to maximize dynamic range
// ============================================================================

// Scale factors for different tensor types (stored on device)
// These allow Q1.15 values to represent larger ranges: actual_value = q115_value * scale
// Default scales chosen to match typical GPT-2 activation ranges

// Embedding scale: token embeddings typically in [-0.1, 0.1] after init
#define Q115_EMBEDDING_SCALE 0.25f

// Attention scale: Q, K, V, attention outputs
#define Q115_ATTENTION_SCALE 1.0f

// FFN scale: feed-forward network activations (can be larger after GELU)
#define Q115_FFN_SCALE 2.0f

// Logits scale: logits before softmax (can be large for confident predictions)
// This is CRITICAL - logits need range of roughly [-10, 10] for proper training
// INCREASED from 16.0 to 24.0 for better softmax expressivity (reduces entropy)
#define Q115_LOGITS_SCALE 24.0f

// Gradient scales: gradients need their own scaling to avoid vanishing
#define Q115_GRAD_EMBEDDING_SCALE 0.01f
#define Q115_GRAD_ATTENTION_SCALE 0.1f
#define Q115_GRAD_FFN_SCALE 0.1f
#define Q115_GRAD_LOGITS_SCALE 1.0f

// ============================================================================
// Q1.15 Optimization Constants
// ============================================================================

// Q-aware LayerNorm epsilon: default 1e-5 is below Q1.15 resolution (~3e-5)
// Using 1e-3 prevents rstd explosion and improves gradient signal
#define Q115_LAYERNORM_EPS 1e-3f

// Residual branch scaling to prevent amplitude collapse
// Attention residual: scale the attention output before adding to residual
// MLP residual: scale the MLP output before adding to residual
#define Q115_ATTENTION_RESIDUAL_SCALE 0.65f
#define Q115_MLP_RESIDUAL_SCALE 0.75f

// Activation clamping range after LayerNorm/RMSNorm
// Prevents rare spikes from poisoning fixed-point math
#define Q115_ACTIVATION_CLAMP_MIN -3.0f
#define Q115_ACTIVATION_CLAMP_MAX 3.0f

// Embedding initialization scales (static, applied at init only)
// Rebalances representational budget between token and position embeddings
#define Q115_WTE_INIT_SCALE 1.2f
#define Q115_WPE_INIT_SCALE 0.8f

// QKV initialization scale for improved head diversity
#define Q115_QKV_INIT_SCALE 1.3f

// Position embedding freeze threshold (freeze wpe after this many steps)
#define Q115_WPE_FREEZE_STEP 3000

// Layer-wise gradient scaling to prevent vanishing gradients in deep networks
// Earlier layers get larger gradients (gradient reversal of the vanishing problem)
__device__ __forceinline__ float layer_gradient_scale(int layer, int total_layers) {
    // Scale gradients inversely with layer depth
    // Layer 0 (closest to output) gets scale 1.0
    // Earlier layers get progressively larger scales
    float depth_ratio = (float)(total_layers - layer) / (float)total_layers;
    return 1.0f + depth_ratio * 0.5f;  // Range: [1.0, 1.5]
}

// ----------------------------------------------------------------------------
// Scaled Q1.15 Conversion Functions

// Convert float to Q1.15 with explicit scale factor
__device__ __forceinline__ q115_t float_to_q115_scaled(float x, float scale) {
    // Normalize by scale factor first
    float normalized = x / scale;
    // Clamp to valid range
    normalized = fmaxf(-Q115_OVERFLOW_THRESHOLD, fminf(Q115_OVERFLOW_THRESHOLD, normalized));
    // Scale and round to nearest integer
    float scaled = normalized * Q115_SCALE;
    int32_t rounded = __float2int_rn(scaled);
    return (q115_t)max(Q115_MIN, min(Q115_MAX, rounded));
}

// Convert Q1.15 to float with explicit scale factor
__device__ __forceinline__ float q115_to_float_scaled(q115_t x, float scale) {
    return (__int2float_rn(x) / Q115_SCALE) * scale;
}

// ----------------------------------------------------------------------------
// Standard Q1.15 Conversion Functions (scale = 1.0)

// Convert float to Q1.15 with clamping
__device__ __forceinline__ q115_t float_to_q115(float x) {
    // Clamp input to valid range
    x = fmaxf(-Q115_OVERFLOW_THRESHOLD, fminf(Q115_OVERFLOW_THRESHOLD, x));
    // Scale and round to nearest integer
    float scaled = x * Q115_SCALE;
    int32_t rounded = __float2int_rn(scaled);
    // Additional safety clamp
    return (q115_t)max(Q115_MIN, min(Q115_MAX, rounded));
}

// Convert Q1.15 to float
__device__ __forceinline__ float q115_to_float(q115_t x) {
    return __int2float_rn(x) / Q115_SCALE;
}

// ----------------------------------------------------------------------------
// Q1.15 Arithmetic Operations

// Q1.15 multiplication: (a * b) >> 15
// Uses 32-bit intermediate result to prevent overflow
__device__ __forceinline__ q115_t q115_mul(q115_t a, q115_t b) {
    int32_t result = ((int32_t)a * (int32_t)b) >> 15;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 addition with saturation
__device__ __forceinline__ q115_t q115_add(q115_t a, q115_t b) {
    int32_t result = (int32_t)a + (int32_t)b;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 subtraction with saturation
__device__ __forceinline__ q115_t q115_sub(q115_t a, q115_t b) {
    int32_t result = (int32_t)a - (int32_t)b;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 negation with saturation
__device__ __forceinline__ q115_t q115_neg(q115_t a) {
    // Special case: -(-1.0) saturates to max value
    if (a == Q115_MIN) return Q115_MAX;
    return -a;
}

// ----------------------------------------------------------------------------
// Scaled Q1.15 Operations (for mixed-scale computations)

// Multiply two Q1.15 values with different scales, return float
__device__ __forceinline__ float q115_mul_scaled(q115_t a, float scale_a, q115_t b, float scale_b) {
    return q115_to_float_scaled(a, scale_a) * q115_to_float_scaled(b, scale_b);
}

// Add two Q1.15 values with different scales, store with output scale
__device__ __forceinline__ q115_t q115_add_scaled(q115_t a, float scale_a, q115_t b, float scale_b, float scale_out) {
    float sum = q115_to_float_scaled(a, scale_a) + q115_to_float_scaled(b, scale_b);
    return float_to_q115_scaled(sum, scale_out);
}

// ----------------------------------------------------------------------------
// Vectorized Q1.15 Operations (for performance)

// Load 2 Q1.15 values as int32_t
__device__ __forceinline__ int32_t load_q115x2(const q115_t* ptr) {
    return *reinterpret_cast<const int32_t*>(ptr);
}

// Store 2 Q1.15 values from int32_t
__device__ __forceinline__ void store_q115x2(q115_t* ptr, int32_t val) {
    *reinterpret_cast<int32_t*>(ptr) = val;
}

// Load 4 Q1.15 values as int64_t
__device__ __forceinline__ int64_t load_q115x4(const q115_t* ptr) {
    return *reinterpret_cast<const int64_t*>(ptr);
}

// Store 4 Q1.15 values from int64_t
__device__ __forceinline__ void store_q115x4(q115_t* ptr, int64_t val) {
    *reinterpret_cast<int64_t*>(ptr) = val;
}

// ----------------------------------------------------------------------------
// Helper Functions

// Check if value will overflow in Q1.15 given a scale
__device__ __forceinline__ bool q115_will_overflow_scaled(float x, float scale) {
    float normalized = x / scale;
    return (normalized >= 1.0f) || (normalized <= -1.0f);
}

// Check if value will overflow in Q1.15
__device__ __forceinline__ bool q115_will_overflow(float x) {
    return (x >= 1.0f) || (x <= -1.0f);
}

// Saturating conversion with overflow detection
__device__ __forceinline__ q115_t float_to_q115_saturate(float x, int* overflow_count = nullptr) {
    bool overflow = q115_will_overflow(x);
    if (overflow && overflow_count != nullptr) {
        atomicAdd(overflow_count, 1);
    }
    return float_to_q115(x);
}

// Adaptive scale computation: finds optimal scale for a value range
__device__ __forceinline__ float compute_optimal_scale(float max_abs_value) {
    // Scale such that max value uses ~75% of Q1.15 range (leave headroom)
    if (max_abs_value < 1e-6f) return 1.0f;  // Avoid division by zero
    return max_abs_value / 0.75f;
}

// ----------------------------------------------------------------------------
// Mixed Precision Operations (Q1.15 with float accumulation)

// Multiply-accumulate: acc += a * b (Q1.15 inputs, float accumulator)
__device__ __forceinline__ void q115_mac_float(float& acc, q115_t a, q115_t b) {
    acc += q115_to_float(a) * q115_to_float(b);
}

// Scaled multiply-accumulate
__device__ __forceinline__ void q115_mac_float_scaled(float& acc, q115_t a, float scale_a, q115_t b, float scale_b) {
    acc += q115_to_float_scaled(a, scale_a) * q115_to_float_scaled(b, scale_b);
}

// Dot product helper (Q1.15 vectors, float result)
__device__ __forceinline__ float q115_dot_product(const q115_t* a, const q115_t* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += q115_to_float(a[i]) * q115_to_float(b[i]);
    }
    return sum;
}

// Scaled dot product
__device__ __forceinline__ float q115_dot_product_scaled(const q115_t* a, float scale_a, 
                                                          const q115_t* b, float scale_b, int n) {
    float sum = 0.0f;
    float combined_scale = scale_a * scale_b;
    for (int i = 0; i < n; i++) {
        sum += q115_to_float(a[i]) * q115_to_float(b[i]);
    }
    return sum * combined_scale;
}

#endif // Q115_COMMON_CUH
