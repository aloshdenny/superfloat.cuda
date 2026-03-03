/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#if defined(ENABLE_Q115)
#include "q115_common.cuh"
#elif defined(ENABLE_Q131)
#include "q131_common.cuh"
#endif

// M_PI is not defined by default on MSVC
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
#if defined(ENABLE_Q131)
        float xi = q131_to_float(packed_inp[k]);
#elif defined(ENABLE_Q115)
        float xi = q115_to_float(packed_inp[k]);
#else
        float xi = (float)packed_inp[k];
#endif
        float cube = 0.044715f * xi * xi * xi;
        float result = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
#if defined(ENABLE_Q131)
        packed_out[k] = float_to_q131(result);
#elif defined(ENABLE_Q115)
        packed_out[k] = float_to_q115(result);
#else
        packed_out[k] = (floatX)result;
#endif
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
#if defined(ENABLE_Q131)
        float x = q131_to_float(packed_inp[k]);
        float dout = q131_to_float(packed_dout[k]);
#elif defined(ENABLE_Q115)
        float x = q115_to_float(packed_inp[k]);
        float dout = q115_to_float(packed_dout[k]);
#else
        float x = (float)packed_inp[k];
        float dout = (float)packed_dout[k];
#endif
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        float result = local_grad * dout;
#if defined(ENABLE_Q131)
        packed_dinp[k] = float_to_q131(result);
#elif defined(ENABLE_Q115)
        packed_dinp[k] = float_to_q115(result);
#else
        packed_dinp[k] = (floatX)result;
#endif
    }
    store128(d_in_out + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}
