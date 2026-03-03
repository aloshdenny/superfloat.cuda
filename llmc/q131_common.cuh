#pragma once

// Q1.31 helpers (signed int32 fixed-point representing [-1, 1))
// This header is intentionally lightweight: conversions + safe multiply.

#include <stdint.h>
#include <cuda_runtime.h>

typedef int32_t q131_t;

// LayerNorm epsilon for Q1.31 paths (kept small; LN math is still float).
#ifndef Q131_LAYERNORM_EPS
#define Q131_LAYERNORM_EPS 1e-5f
#endif

// Scale factor for mapping float in [-1,1) to Q1.31 integer.
// Note: 2^31 does not fit in int32, so we use int64 constants.
constexpr int64_t Q131_SCALE_I64 = (1LL << 31);
constexpr double  Q131_SCALE_D   = 2147483648.0; // 2^31

__device__ __forceinline__ float q131_to_float(q131_t v) {
    return (float)((double)v / Q131_SCALE_D);
}

// Forward declaration (some kernels use float_to_q131(), which aliases to this).
__device__ __forceinline__ q131_t float_to_q131_rne(float x);

// Historical name used in kernels.
__device__ __forceinline__ q131_t float_to_q131(float x) {
    return float_to_q131_rne(x);
}

// Saturating add (Q1.31 + Q1.31 -> Q1.31).
__device__ __forceinline__ q131_t q131_add(q131_t a, q131_t b) {
    int64_t s = (int64_t)a + (int64_t)b;
    if (s > 0x7fffffffLL) s = 0x7fffffffLL;
    if (s < (long long)0x80000000LL) s = (long long)0x80000000LL;
    return (q131_t)s;
}

// Saturating subtract.
__device__ __forceinline__ q131_t q131_sub(q131_t a, q131_t b) {
    int64_t s = (int64_t)a - (int64_t)b;
    if (s > 0x7fffffffLL) s = 0x7fffffffLL;
    if (s < (long long)0x80000000LL) s = (long long)0x80000000LL;
    return (q131_t)s;
}

// Round-to-nearest-even (RNE) conversion.
// Clamps to representable Q1.31 range: [-2^31, 2^31-1].
__device__ __forceinline__ q131_t float_to_q131_rne(float x) {
    // Clamp to [-1, 1) in float domain to avoid overflow.
    // Largest representable positive is (2^31-1)/2^31.
    if (x >= 1.0f) {
        return (q131_t)0x7fffffff;
    }
    if (x <= -1.0f) {
        return (q131_t)0x80000000;
    }

    // Use double for higher precision prior to rounding.
    double scaled = (double)x * Q131_SCALE_D;
    long long q = __double2ll_rn(scaled); // RNE

    if (q > 0x7fffffffLL) q = 0x7fffffffLL;
    if (q < (long long)0x80000000LL) q = (long long)0x80000000LL;
    return (q131_t)q;
}

// Multiply two Q1.31 values with int64 MAC and return Q1.31.
__device__ __forceinline__ q131_t q131_mul(q131_t a, q131_t b) {
    int64_t prod = (int64_t)a * (int64_t)b; // Q2.62
    // Round before shifting back to Q1.31.
    // Add 0.5 ulp at bit 30 (since shifting by 31).
    int64_t rounded = prod + (1LL << 30);
    int64_t shifted = rounded >> 31;
    if (shifted > 0x7fffffffLL) shifted = 0x7fffffffLL;
    if (shifted < (long long)0x80000000LL) shifted = (long long)0x80000000LL;
    return (q131_t)shifted;
}
