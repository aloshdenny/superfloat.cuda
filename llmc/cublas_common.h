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

// 256 MiB workspace — gives cuBLASLt room to find algorithms for all shapes.
// Memory is available now that master weights are disabled.
const size_t cublaslt_workspace_size = 256 * 1024 * 1024;
void *cublaslt_workspace = NULL;
// CUBLAS_COMPUTE_32F_FAST_16BF: uses BF16 Tensor Cores natively.
// Widest algorithm coverage for BF16 inputs on SM 8.9 (RTX 4090).
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F_FAST_16BF;
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