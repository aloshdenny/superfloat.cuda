#pragma once
#include "cuda_common.h"
#include "q115_common.cuh"

#if defined(ENABLE_Q115)
inline float* get_q115_workspace_A(size_t bytes) {
    static float* ptr = NULL; static size_t current_size = 0;
    if (bytes > current_size) { if(ptr) cudaFree(ptr); cudaMalloc(&ptr, bytes); current_size = bytes; }
    return ptr;
}
inline float* get_q115_workspace_B(size_t bytes) {
    static float* ptr = NULL; static size_t current_size = 0;
    if (bytes > current_size) { if(ptr) cudaFree(ptr); cudaMalloc(&ptr, bytes); current_size = bytes; }
    return ptr;
}
inline float* get_q115_workspace_C(size_t bytes) {
    static float* ptr = NULL; static size_t current_size = 0;
    if (bytes > current_size) { if(ptr) cudaFree(ptr); cudaMalloc(&ptr, bytes); current_size = bytes; }
    return ptr;
}
inline float* get_q115_workspace_Bias(size_t bytes) {
    static float* ptr = NULL; static size_t current_size = 0;
    if (bytes > current_size) { if(ptr) cudaFree(ptr); cudaMalloc(&ptr, bytes); current_size = bytes; }
    return ptr;
}

__global__ void q115_to_fp32_kernel(float* dst, const floatX* src, size_t N) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = q115_to_float(src[idx]);
}
__global__ void fp32_to_q115_kernel(floatX* dst, const float* src, size_t N) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    // We optionally multiply by a scale here or just clamp
    if (idx < N) dst[idx] = float_to_q115(fmaxf(-0.999f, fminf(0.999f, src[idx])));
}
__global__ void fp32_to_q115_accumulate_kernel(floatX* dst, const float* src, size_t N) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < N) {
        float existing = q115_to_float(dst[idx]);
        dst[idx] = float_to_q115(fmaxf(-0.999f, fminf(0.999f, src[idx] + existing)));
    }
}

#endif
