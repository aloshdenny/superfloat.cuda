/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <cublasLt.h>
#include <cublas_v2.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#elif defined(ENABLE_Q115)
// Q1.15 forward is simulated at tensor boundaries while storage remains BF16.
// Keep low-precision tensor layout aligned with floatX storage type.
#define CUBLAS_LOWP CUDA_R_16BF
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

#ifndef CUBLAS_COMPUTE_32F_FAST_16BF
#define CUBLAS_COMPUTE_32F_FAST_16BF CUBLAS_COMPUTE_32F
#endif

// Keep workspace small to avoid OOM on memory-constrained runs.
// 4 MiB is more than enough for cuBLASLt to find Tensor Core BF16 algorithms.
const size_t cublaslt_workspace_size = 4 * 1024 * 1024;
void *cublaslt_workspace = NULL;
// CUBLAS_COMPUTE_32F_FAST_TF32: accumulate in TF32 precision via Tensor Cores.
// This is the correct compute type for BF16 inputs on SM 8.9 (RTX 4090).
// CUBLAS_COMPUTE_32F (exact FP32) has no BF16 Tensor Core path in cuBLASLt.
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F_FAST_TF32;
cublasLtHandle_t cublaslt_handle;
cublasHandle_t cublas_handle;  // for plain cublasGemmEx calls (LLaMA forward)

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status)                                                    \
  {                                                                            \
    cublasCheck((status), __FILE__, __LINE__);                                 \
  }

#endif // CUBLAS_COMMON_H