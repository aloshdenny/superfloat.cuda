/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits> // std::bool_constant
// llmc internal imports
#include "cublas_common.h"
#include "cuda_common.h"
#include "cuda_utils.cuh"
// GELU can be either fused (cublasLt) or non-fused (gelu.h)
#include "gelu.cuh"

// Define missing cuBLASLt constants that might not be in older CUDA versions
#ifndef CUBLASLT_EPILOGUE_DGELU
#define CUBLASLT_EPILOGUE_DGELU (cublasLtEpilogue_t)(64 | 128)
#endif

#ifndef CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
#define CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE (cublasLtMatmulDescAttributes_t)26
#endif

// Define missing cuBLASLt constants that might not be in older CUDA versions
#ifndef CUBLASLT_EPILOGUE_DGELU
#define CUBLASLT_EPILOGUE_DGELU (cublasLtEpilogue_t)(64 | 128)
#endif

#ifndef CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
#define CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE 26
#endif
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
#if defined(SF16_TRUE_FORWARD)
#include "q131_common.cuh"
#endif
#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

template <typename OutFloat, bool UseAuxBuffer>
__global__ void
matmul_backward_bias_kernel9(OutFloat *dbias, const floatX *dout, int B, int T,
                             int OC, std::bool_constant<UseAuxBuffer>) {
  constexpr const int bdx = 4;
  constexpr const int bdy = WARP_SIZE / bdx;
  assert(blockDim.x == bdx);
  assert(blockDim.y == bdy);

  int warp_d = (int)threadIdx.x;
  int warp_c = (int)threadIdx.y;
  int block_d = (int)threadIdx.z;

  const int OC_per_warp = bdy * x128::size; // 64 at BF16

  int local_oc = warp_c * x128::size;
  int global_oc = blockIdx.x * OC_per_warp + local_oc;

  int local_bt = warp_d + bdx * block_d;
  int bt_per_block = bdx * blockDim.z;

  float accumulators[x128::size];
  for (int k = 0; k < x128::size; k++) {
    accumulators[k] = 0.0f;
  }

  if (global_oc < OC) {
    // sum up over all bt within registers
    for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T;
         idx += gridDim.y * bt_per_block) {
      x128 packed_dout = load128(dout + global_oc + idx * OC);
      for (int k = 0; k < x128::size; k++) {
        accumulators[k] += (float)packed_dout[k];
      }
    }
  }

  __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

  // reduce within-warp results
  for (int k = 0; k < x128::size; k++) {
    float v = accumulators[k];
    v += __shfl_down_sync(0xffffffff, v, 1, 4);
    v += __shfl_down_sync(0xffffffff, v, 2, 4);
    if (warp_d == 0) {
      sub_results[k][block_d][warp_c] = v;
    }
  }
  __syncthreads();

  // block-wide reductions
  for (int k = block_d; k < x128::size; k += blockDim.z) {
    float a = 0.f;
    for (int r = warp_d; r < blockDim.z; r += bdx) {
      float v = sub_results[k][r][warp_c];
      v += __shfl_down_sync(0xffffffff, v, 1, 4);
      v += __shfl_down_sync(0xffffffff, v, 2, 4);
      a += v;
    }
    if (warp_d == 0 && global_oc < OC) {
      if constexpr (!UseAuxBuffer) {
        dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
      } else {
        dbias[global_oc + k + blockIdx.y * OC] = a;
      }
    }
  }
}

__global__ void reduce_add_sum_kernel(floatX *dst, const float *src, size_t n,
                                      size_t m) {
  const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
  assert(n % x128::size == 0);
  if (idx < n) {
    f128 acc;
    for (int k = 0; k < f128::size; ++k) {
      acc[k] = 0.f;
    }

    for (int l = 0; l < m; ++l) {
      f128 s = load128(src + idx + n * l);
      for (int k = 0; k < f128::size; ++k) {
        acc[k] += s[k];
      }
    }
    for (int k = 0; k < f128::size; ++k) {
      dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
    }
  }
}

// ----------------------------------------------------------------------------
// kernel launchers

#if defined(ENABLE_Q115)
// Quantize forward outputs to SF16 boundaries.
// In SF16_TRUE_FORWARD mode, we first simulate an SF32 accumulator register
// and then quantize to SF16 storage boundaries.
__global__ void q115_simulate_kernel(floatX *d, size_t N) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = (float)d[idx];
#if defined(SF16_TRUE_FORWARD)
    v = simulate_q131(v);
#endif
    d[idx] = (floatX)simulate_q115(v);
  }
}
#endif

// Wrapper around cublasLtMatmul that is meant to support everything we need in
// llm.c https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX *d, const floatX *a, const floatX *b,
                     const floatX *bias, int m, int n, int k,
                     cudaStream_t stream = 0, bool transA = true,
                     bool transB = false, int batch_count = 0,
                     size_t strideA = 0, size_t strideB = 0,
                     size_t strideOut = 0, bool accumulate = false,
                     floatX *pre_gelu = NULL, bool backward = false) {
  NVTX_RANGE_FN();
  bool has_bias = (bias != NULL);
  bool has_gelu = (pre_gelu != NULL);
#if defined(ENABLE_Q115)
  size_t size_C = (size_t)batch_count > 0 ? (size_t)batch_count * strideOut
                                          : (size_t)m * n;
  if (size_C == 0) {
    size_C = (size_t)m * n;
  }
#endif

  // check alignment (only for non-null pointers)
  if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 ||
      ((uintptr_t)d % 16) != 0 || (has_bias && ((uintptr_t)bias % 16) != 0)) {
    printf("All cuBLASLt pointers must be aligned!\n");
    exit(EXIT_FAILURE);
  }

  // create the operation descriptor
  cublasLtMatmulDesc_t operationDesc;
  cublasCheck(
      cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

  int returnedResults = 0;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;

  cublasOperation_t opNoTranspose = CUBLAS_OP_N;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasCheck(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
      (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
      (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

  // define matrix layouts
  cublasLtMatrixLayout_t ALayout;
  cublasLtMatrixLayout_t BLayout;
  cublasLtMatrixLayout_t DLayout;
  cublasLtMatrixLayout_t CLayout;
  if (transA) {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
  }
  if (transB) {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
  }
  // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
  cublasCheck(cublasLtMatrixLayoutCreate(
      &CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
  cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

  // Strided Batched GEMM (used for non-flash attention, equivalent to
  // cublasGemmStridedBatchedEx)
  if (batch_count) {
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));

    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
        sizeof(strideA)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
        sizeof(strideB)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut,
        sizeof(strideOut)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut,
        sizeof(strideOut)));
  }

  // create a preference handle with specified max workspace
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // setup epilogue and associated pointers for bias & gelu
  cublasLtEpilogue_t epilogue;
  if (has_gelu) {
    int64_t gelu_ld = m; // todo - is this affected by anything else?
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld,
        sizeof(gelu_ld)));
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu,
        sizeof(pre_gelu)));
    if (backward) {
      assert(!has_bias); // we shouldn't have any backward matmuls that use both
                         // GELU and bias
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS
                          : CUBLASLT_EPILOGUE_GELU_AUX;
    }
  } else if (has_bias) {
    epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
  } else {
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  }
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
                                             CUBLASLT_MATMUL_DESC_EPILOGUE,
                                             &epilogue, sizeof(epilogue)));

  if (has_bias) {
    // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
    cublasDataType_t bias_data_type =
        (sizeof(floatX) == 1) ? CUDA_R_16BF
                              : CUBLAS_LOWP; // force BF16 bias for FP8 mode
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type,
        sizeof(bias_data_type)));
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }

  // set scale type to FP32 (needs to be FP16 if and only if using
  // CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
  cublasDataType_t scale_type = CUDA_R_32F;
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
                                             CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                             &scale_type, sizeof(scale_type)));
  // find a suitable algorithm (same as GPT-2)
  cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout,
                                 BLayout, CLayout, DLayout, preference, 1,
                                 &heuristic, &returnedResults);
  if (returnedResults == 0) {
    printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
    exit(EXIT_FAILURE);
  }

  // set whether to accumulate (i.e. D += C) or not - note this isn't considered
  // in algorithm selection (?!)
  const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

  // call the matmul natively on bfloat16
  cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, a, ALayout,
                             b, BLayout, &beta, d, CLayout, d, DLayout,
                             &heuristic.algo, cublaslt_workspace,
                             cublaslt_workspace_size, stream));

#if defined(ENABLE_Q115)
  // For Q1.15 forward simulation, restrict outputs to valid SF16 bounds.
  // In SF16_TRUE_FORWARD mode this also injects an SF32 register simulation
  // before the SF16 write-back quantization.
  if (!backward && size_C > 0) {
    int num_blocks = (size_C + 255) / 256;
    q115_simulate_kernel<<<num_blocks, 256, 0, stream>>>(d, size_C);
  }
#endif
  // cleanups
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
  cudaCheck(cudaGetLastError());
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical
// order of arguments)
void matmul_forward_cublaslt(floatX *out, floatX *inp, floatX *weight,
                             floatX *bias, int B, int T, int C, int OC,
                             cudaStream_t stream, floatX *pre_gelu = NULL,
                             int gelu_fusion = 1) {
  // By default only fuse GELU for H100+ as cuBLAS seems to be inefficient for
  // fused GELU on Ada/Ampere (?)
  if (gelu_fusion < 1 && pre_gelu) {
    matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B * T, C, stream, true,
                    false, 0, 0, 0, 0, false, NULL, false);
    gelu_forward(out, pre_gelu, B * T * OC, stream);
  } else {
    matmul_cublaslt(out, weight, inp, bias, OC, B * T, C, stream, true, false,
                    0, 0, 0, 0, false, pre_gelu, false);
  }
}

// ---------------------------------------------------------------------------
// Forward GEMM for LLaMA (no bias).
// cuBLASLt fails to find algorithms for LLaMA's matrix dimensions with ANY
// compute type or epilogue on this system. cublasSgemm (plain FP32) always
// works. Cast BF16/FP16 → FP32, run SGEMM, cast output FP32 → BF16/FP16.
// ---------------------------------------------------------------------------
static float *g_sgemm_A = nullptr;   // FP32 copy of weight [OC x C]
static float *g_sgemm_B = nullptr;   // FP32 copy of inp    [BT x C]
static float *g_sgemm_C = nullptr;   // FP32 output         [OC x BT]
static size_t g_sgemm_A_size = 0;
static size_t g_sgemm_B_size = 0;
static size_t g_sgemm_C_size = 0;

__global__ void upcast_to_fp32(float *dst, const floatX *src, size_t N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dst[i] = (float)src[i];
}
__global__ void downcast_from_fp32(floatX *dst, const float *src, size_t N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dst[i] = (floatX)src[i];
}

void matmul_forward_cublas(floatX *out, const floatX *inp, const floatX *weight,
                           int B, int T, int C, int OC, cudaStream_t stream) {
  NVTX_RANGE_FN();
  size_t szA = (size_t)OC * C;           // weight: OC x C
  size_t szB = (size_t)B * T * C;        // inp:    BT x C
  size_t szC = (size_t)OC * B * T;       // out:    OC x BT

  // Lazy-grow FP32 scratch buffers
  if (szA > g_sgemm_A_size) { if (g_sgemm_A) cudaFree(g_sgemm_A); cudaCheck(cudaMalloc(&g_sgemm_A, szA*sizeof(float))); g_sgemm_A_size = szA; }
  if (szB > g_sgemm_B_size) { if (g_sgemm_B) cudaFree(g_sgemm_B); cudaCheck(cudaMalloc(&g_sgemm_B, szB*sizeof(float))); g_sgemm_B_size = szB; }
  if (szC > g_sgemm_C_size) { if (g_sgemm_C) cudaFree(g_sgemm_C); cudaCheck(cudaMalloc(&g_sgemm_C, szC*sizeof(float))); g_sgemm_C_size = szC; }

  // Cast inputs to FP32
  upcast_to_fp32<<<CEIL_DIV(szA, 256), 256, 0, stream>>>(g_sgemm_A, weight, szA);
  upcast_to_fp32<<<CEIL_DIV(szB, 256), 256, 0, stream>>>(g_sgemm_B, inp,    szB);
  cudaCheck(cudaGetLastError());

  // SGEMM: C = A^T * B  →  out[OC x BT] = weight[OC x C]^T-nope, weight stored as [OC,C]
  // Column-major view: weight is (C x OC) with ld=C, inp is (C x BT) with ld=C
  // cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  // We want: out(OC,BT) = weight(OC,C) @ inp(C,BT)
  // In col-major: weight col-major is (C rows, OC cols) with lda=C → CUBLAS_OP_T gives OC x C
  // Simpler: out = weight * inp^T, no: out[OC,BT] = weight[OC,C] @ inp[BT,C]^T
  // Col-major equiv: op(A)=weight^T[C,OC], op(B)=inp[C,BT], C=out[OC,BT]
  //   → SGEMM with transA=T m=OC n=BT k=C A=weight lda=C B=inp ldb=C C=out ldc=OC
  cublasCheck(cublasSetStream(cublas_handle, stream));
  const float alpha = 1.f, beta = 0.f;
  cublasCheck(cublasSgemm(
      cublas_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,   // weight^T, inp (no transpose)
      OC, B * T, C,               // m=OC, n=BT, k=C
      &alpha,
      g_sgemm_A, C,               // A=weight(C cols, OC rows in col-major), lda=C
      g_sgemm_B, C,               // B=inp(C rows, BT cols in col-major), ldb=C
      &beta,
      g_sgemm_C, OC               // C=out(OC rows, BT cols), ldc=OC
  ));

  // Cast FP32 output back to floatX
  downcast_from_fp32<<<CEIL_DIV(szC, 256), 256, 0, stream>>>(out, g_sgemm_C, szC);
  cudaCheck(cudaGetLastError());
}

void matmul_backward(floatX *dinp, floatX *dweight, floatX *dbias, floatX *dout,
                     floatX *inp, floatX *weight, float *dbias_buffer, int B,
                     int T, int C, int OC, cudaStream_t stream,
                     floatX *pre_gelu = NULL, int gelu_fusion = 1) {
  NVTX_RANGE_FN();

  // backward to bias, if given, does a +=
  if (dbias != NULL) {
    // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must
    // be a multiple of 64!) Block size is 1024 | 768 threads (32|24 warps) and
    // we reduce those values into 1 at the end

    const int block_size =
        deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

    dim3 block_dim = {4, 8, (unsigned)block_size / WARP_SIZE};
    const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
    const int grid_size_x = CEIL_DIV(
        OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor *
                                       deviceProp.multiProcessorCount /
                                       (block_size * grid_size_x)); // full GPU!

    // If we have enough OC that we don't need cross-block reductions, we can
    // skip the bias_buffer accumulation and write results directly to the
    // output.
    if (grid_size_y == 1) {
      matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim,
                                     0, stream>>>(dbias, dout, B, T, OC, False);
      cudaCheck(cudaGetLastError());
    } else {
      // kernel 9 overwrites temp buffer, so no need to memset
      matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim,
                                     0, stream>>>(dbias_buffer, dout, B, T, OC,
                                                  True);
      cudaCheck(cudaGetLastError());
      reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(
          dbias, dbias_buffer, OC, grid_size_y);
      cudaCheck(cudaGetLastError());
    }
    dbias = NULL; // prevent dbias calculation from also being fused in
                  // matmul_cublaslt below (if we enabled fusion)
  }

  // backward to input, uses = in the backward pass (set the gradient)
  matmul_cublaslt(dinp, weight, dout, NULL, C, B * T, OC, stream, false, false,
                  0, 0, 0, 0, false, gelu_fusion >= 2 ? pre_gelu : NULL, true);

  // backward GELU (if it wasn't fused into the matmul above)
  if (gelu_fusion < 2 && pre_gelu) {
    gelu_backward_inplace(dinp, pre_gelu, B * T * C, stream);
  }

  // backward to weight, uses += in the backward pass (accumulate the gradient)
  // by setting alpha=one
  matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B * T, stream,
                  false, true, 0, 0, 0, 0, true /* accumulate */, NULL, true);
}
