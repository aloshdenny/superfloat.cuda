#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define cublasCheck(stmt) do {                                 \
    cublasStatus_t err = stmt;                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                        \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                               \
    }                                                          \
} while(0)

int main() {
    cublasLtHandle_t handle;
    cublasCheck(cublasLtCreate(&handle));

    int m = 3072;
    int n = 2048;
    int k = 2048;

    cublasLtMatmulDesc_t opDesc;
    cublasCheck(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasCheck(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    cublasCheck(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t ALayout, BLayout, CLayout;
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_16BF, k, m, k));
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_16BF, k, n, k));
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_16BF, m, n, m));

    size_t workspace_size = 64 * 1024 * 1024;
    cublasLtMatmulPreference_t pref;
    cublasCheck(cublasLtMatmulPreferenceCreate(&pref));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(handle, opDesc, ALayout, BLayout, CLayout, CLayout, pref, 1, &heuristic, &returnedResults);

    if (returnedResults == 0) {
        printf("No algos found for compute type CUBLAS_COMPUTE_32F_FAST_16BF and input CUDA_R_16BF\n");
    } else {
        printf("Success!\n");
    }
}
