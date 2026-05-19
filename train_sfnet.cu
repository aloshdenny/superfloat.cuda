/*
SFNet — a Transformer architecture designed specifically for strict SF16
(Q1.15) forward storage in the range [-0.999969..., +0.999969...] with
normal BF16/FP backward.

Why a new architecture?
======================
GPT-2 is a 2019 architecture (LayerNorm with biases, learned absolute
positional embeddings, MHA with vanilla softmax, GELU, untied LM head,
two residual additions per layer).  Each of these is hostile to Q1.15
storage:

  * Biases drift outside [-1, 1).
  * Absolute position embeddings can grow unboundedly with sequence length.
  * Vanilla softmax can produce 1.0-eps values that flush all competing
    rows to 0 in Q1.15 quantization.
  * GELU(x) is unbounded above.
  * Each residual addition is a Q1.15 saturation site.
  * Untied LM head doubles the number of Q1.15-quantized embedding rows.

SFNet replaces every one of these with a Q1.15-friendly choice while
maintaining or improving expressivity, and is laid out so the CUDA path
is strictly contractive at every kernel boundary.

Per layer (PaLM-style parallel attention + MLP block):

  x_norm   = RMSNorm(x, rms1w)
  ── attention branch ──
    qkv      = qkvw @ x_norm                   (compact GQA layout)
    qkv.Q    = QHeadRMSNorm(qkv.Q, q_norm_w)
    qkv.K    = QHeadRMSNorm(qkv.K, k_norm_w)
    qkv.Q,K  = RoPE(qkv.Q,K)
    qkv_full = expand_gqa(qkv)
    a        = softmax_attention(qkv_full)     (Q1.15-stable thanks to QK-Norm)
    attn_out = attn_ow @ a
  ── MLP branch (uses the SAME x_norm) ──
    gate     = gate_w @ x_norm
    up       = up_w   @ x_norm
    glu      = tanh(gate) * up
    mlp_out  = down_w @ glu
  ── single norm-preserving residual ──
    x' = sqrt(1 - α²) * x + α * (attn_out + mlp_out)

Final RMSNorm + tied LM head:
    z      = RMSNorm(x_L, rms_fw)
    logits = wte^T @ z

Build:
    make train_sfnetcu                 # BF16 baseline
    make train_sfnetq115cu             # strict SF16 forward / BF16 backward

Run:
    ./train_sfnetq115cu -e sfnet:c768 -b 4 -t 1024 -x 10000

Model variants:
    sfnet:c512   ~30M
    sfnet:c768   ~125M   (default, GPT-2 124M parity)
    sfnet:c1024  ~330M
    sfnet:c1536  ~750M
    sfnet:c2048  ~1.3B
*/

// ============================================================================
// Platform / OS headers
// ============================================================================
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#include <direct.h>
#include <io.h>
#include <windows.h>
#define access _access
#ifndef F_OK
#define F_OK 0
#endif
#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif
#else
#include <unistd.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>

// ============================================================================
// CPU utilities
// ============================================================================
#include "llmc/utils.h"
#include "llmc/tokenizer.h"
#include "llmc/dataloader.h"
#include "llmc/rand.h"
#include "llmc/schedulers.h"
#include "llmc/sampler.h"
#include "llmc/logger.h"
#include "llmc/mfu.h"
#include "llmc/outlier_detector.h"

// ============================================================================
// GPU utilities
// ============================================================================
#include "llmc/cuda_common.h"
#if defined(ENABLE_Q115)
#include "llmc/q115_common.cuh"
#if defined(SF16_TRUE_FORWARD)
#include "llmc/q131_common.cuh"
#endif
#endif
#include "llmc/cuda_utils.cuh"
#include "llmc/cublas_common.h"

// ============================================================================
// Layer implementations (existing + SFNet-specific)
// ============================================================================
#include "llmc/encoder.cuh"
#include "llmc/rmsnorm.cuh"
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"
#include "llmc/attention.cuh"
#include "llmc/fused_classifier.cuh"
#include "llmc/adamw.cuh"
#include "llmc/global_norm.cuh"
#include "llmc/zero.cuh"
#include "llmc/sfnet_layers.cuh"

// ============================================================================
// Globals
// ============================================================================
char filename_buffer[512];
cudaDeviceProp deviceProp;
cudaStream_t main_stream;
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;
static unsigned int g_last_sanitized_grad_count = 0;

// ============================================================================
// Local matmul wrappers — thin shims over matmul_forward_cublaslt / matmul_cublaslt
// ----------------------------------------------------------------------------
// cuBLASLt requires a power-of-2 (or otherwise "canonical") OC for its
// heuristic to return algorithms.  We enforce this via sfnet_set_hyperparameters
// which rounds qkv_w up to the next power of 2.  All other OC values used by
// SFNet (C=768, FFN=2048, Vp=50304) are already cuBLASLt-friendly.
//
// The accumulate_dinp flag in sfnet_matmul_backward is needed because gate, up,
// and QKV all accumulate their dinp contribution into the same dl_rms1 buffer.
// ============================================================================
static inline void sfnet_matmul_forward(floatX *out,
                                        const floatX *inp, const floatX *weight,
                                        int B, int T, int C, int OC,
                                        cudaStream_t stream,
                                        bool is_logits = false) {
    matmul_forward_cublaslt(out, (floatX *)inp, (floatX *)weight, /*bias=*/nullptr,
                            B, T, C, OC, stream, /*pre_gelu=*/nullptr,
                            /*gelu_fusion=*/1, is_logits);
}

static inline void sfnet_matmul_backward(floatX *dinp, floatX *dweight,
                                         floatX *dout,  floatX *inp, floatX *weight,
                                         int B, int T, int C, int OC,
                                         bool accumulate_dinp, cudaStream_t stream) {
    if (dinp) {
        matmul_cublaslt(dinp, weight, dout, /*bias=*/nullptr,
                        C, B * T, OC, stream,
                        /*transA=*/false, /*transB=*/false,
                        0, 0, 0, 0, accumulate_dinp, nullptr, /*backward=*/true);
    }
    if (dweight) {
        matmul_cublaslt(dweight, inp, dout, /*bias=*/nullptr,
                        C, OC, B * T, stream,
                        /*transA=*/false, /*transB=*/true,
                        0, 0, 0, 0, /*accumulate=*/true, nullptr, /*backward=*/true);
    }
}

// ============================================================================
// Config
// ============================================================================
typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int ffn_dim;
    int head_dim;
    int vocab_size;
    int padded_vocab_size;
    int max_seq_len;
    float rope_theta;
    float norm_eps;
    float alpha_init;
} SFNetConfig;

// ============================================================================
// Parameter tensors  (11 tensors total, no biases anywhere)
// ============================================================================
//   0: wte         (Vp, C)               — tied embedding + LM head
//   1: rms1w       (L, C)
//   2: qkvw        (L, (NH+2*NKV)*HD, C)
//   3: q_norm_w    (L, NH, HD)
//   4: k_norm_w    (L, NKV, HD)
//   5: attn_ow     (L, C, NH*HD)
//   6: gate_w      (L, FFN, C)
//   7: up_w        (L, FFN, C)
//   8: down_w      (L, C, FFN)
//   9: alpha       (L,)                  — per-layer scaled-residual α
//  10: rms_fw      (C,)
constexpr const int NUM_PARAMETER_TENSORS = 11;
typedef struct {
    floatX *wte;
    floatX *rms1w;
    floatX *qkvw;
    floatX *q_norm_w;
    floatX *k_norm_w;
    floatX *attn_ow;
    floatX *gate_w;
    floatX *up_w;
    floatX *down_w;
    floatX *alpha;
    floatX *rms_fw;
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void *),
              "ParameterTensors size mismatch!");

void fill_in_parameter_sizes(size_t *param_sizes, size_t *param_sizeof, SFNetConfig cfg) {
    size_t Vp  = (size_t)cfg.padded_vocab_size;
    size_t C   = (size_t)cfg.dim;
    size_t L   = (size_t)cfg.n_layers;
    size_t NH  = (size_t)cfg.n_heads;
    size_t NKV = (size_t)cfg.n_kv_heads;
    size_t HD  = (size_t)cfg.head_dim;
    size_t FFN = (size_t)cfg.ffn_dim;

    param_sizes[0]  = Vp * C;
    param_sizes[1]  = L * C;
    param_sizes[2]  = L * (NH + 2 * NKV) * HD * C;
    param_sizes[3]  = L * NH * HD;
    param_sizes[4]  = L * NKV * HD;
    param_sizes[5]  = L * C * (NH * HD);
    param_sizes[6]  = L * FFN * C;
    param_sizes[7]  = L * FFN * C;
    param_sizes[8]  = L * C * FFN;
    param_sizes[9]  = L;
    param_sizes[10] = C;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) param_sizeof[i] = sizeof(floatX);
}

void *malloc_and_point_parameters(ParameterTensors *params, size_t *param_elements,
                                  size_t *param_sizeof) {
    size_t num_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
        num_bytes += param_elements[i] * param_sizeof[i];

    void *mem;
    cudaCheck(cudaMalloc(&mem, num_bytes));

    floatX **ptrs[] = {
        &params->wte, &params->rms1w, &params->qkvw,
        &params->q_norm_w, &params->k_norm_w, &params->attn_ow,
        &params->gate_w, &params->up_w, &params->down_w,
        &params->alpha, &params->rms_fw
    };
    char *it = (char *)mem;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX *)it;
        it += param_elements[i] * param_sizeof[i];
    }
    return mem;
}

// ============================================================================
// Activation tensors
// ----------------------------------------------------------------------------
// Layout decisions:
//   * residual stream is saved per-layer (L+1 buffers via encoded + residual[L])
//   * x_norm (rms1) and rstd are saved per-layer
//   * QKV pre-norm AND QKV post-norm are saved per-layer
//       — qkv_pre is needed for QK-Norm backward (the pre-norm input)
//       — qkv_post (after QK-Norm + RoPE) is needed because attention_backward
//         reconstructs the attention matrix from Q,K
//   * qkvr_perm (the permuted (3,B,NH,T,HS) buffer written by attention_forward)
//     is saved per-layer
//   * gate, up, glu are saved per-layer (recomputation possible but kept for
//     code simplicity)
//   * att is a SINGLE-LAYER scratch buffer (B*NH*T*T) — backward iterates
//     layer-by-layer so we don't need L copies
//   * output is the big universal scratch / classifier buffer
// ============================================================================
constexpr int NUM_ACTIVATION_TENSORS = 23;
typedef struct {
    floatX *encoded;        // (B, T, C)
    floatX *residual;       // (L, B, T, C)  — x after each block
    floatX *rms1;           // (L, B, T, C)
    float  *rms1_rstd;      // (L, B, T)
    floatX *qkv_pre;        // (L, B, T, qkv_w)
    floatX *qkv_post;       // (L, B, T, qkv_w)
    float  *rstd_q;         // (L, B, T, NH)
    float  *rstd_k;         // (L, B, T, NKV)
    floatX *qkvr_perm;      // (L, 3, B, NH, T, HS) — permuted Q,K,V (attn_forward output)
    floatX *atty;           // (L, B, T, C)
    floatX *att;            // (B, NH, T, T)  — single-layer scratch
    floatX *attn_out;       // (B, T, C)      — per-layer attention output (scratch)
    floatX *gate;           // (L, B, T, FFN)
    floatX *up;             // (L, B, T, FFN)
    floatX *glu;            // (L, B, T, FFN)
    floatX *mlp_out;        // (B, T, C)      — per-layer MLP output (scratch)
    floatX *rms_f;          // (B, T, C)
    float  *rms_f_rstd;     // (B, T)
    float  *losses;         // (B, T)
    floatX *output;         // (B, T, max(qkv_w, Vp)) — huge scratch
    floatX *scratch_btc;    // (B, T, C)
    floatX *scratch_btc2;   // (B, T, C)
    floatX *scratch_qkv3c;  // (B, T, 3*NH*HD) — expanded GQA input to attention
} ActivationTensors;

struct TensorSpec {
    void **ptr;
    size_t size;
    DType type;
};

#define TENSOR_SPEC(pointer, size) \
    TensorSpec{(void **)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors *data,
                               TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS],
                               size_t B, size_t T, SFNetConfig cfg) {
    size_t C   = cfg.dim;
    size_t L   = cfg.n_layers;
    size_t NH  = cfg.n_heads;
    size_t NKV = cfg.n_kv_heads;
    size_t HD  = cfg.head_dim;
    size_t FFN = cfg.ffn_dim;
    size_t Vp  = cfg.padded_vocab_size;
    size_t qkv_w = (NH + 2 * NKV) * HD;

    int i = 0;
    tensors[i++] = TENSOR_SPEC(data->encoded,       B * T * C);
    tensors[i++] = TENSOR_SPEC(data->residual,      L * B * T * C);
    tensors[i++] = TENSOR_SPEC(data->rms1,          L * B * T * C);
    tensors[i++] = TENSOR_SPEC(data->rms1_rstd,     L * B * T);
    tensors[i++] = TENSOR_SPEC(data->qkv_pre,       L * B * T * qkv_w);
    tensors[i++] = TENSOR_SPEC(data->qkv_post,      L * B * T * qkv_w);
    tensors[i++] = TENSOR_SPEC(data->rstd_q,        L * B * T * NH);
    tensors[i++] = TENSOR_SPEC(data->rstd_k,        L * B * T * NKV);
    tensors[i++] = TENSOR_SPEC(data->qkvr_perm,     L * B * T * 3 * C);
    tensors[i++] = TENSOR_SPEC(data->atty,          L * B * T * C);
    tensors[i++] = TENSOR_SPEC(data->att,           B * NH * T * T);
    tensors[i++] = TENSOR_SPEC(data->attn_out,      B * T * C);
    tensors[i++] = TENSOR_SPEC(data->gate,          L * B * T * FFN);
    tensors[i++] = TENSOR_SPEC(data->up,            L * B * T * FFN);
    tensors[i++] = TENSOR_SPEC(data->glu,           L * B * T * FFN);
    tensors[i++] = TENSOR_SPEC(data->mlp_out,       B * T * C);
    tensors[i++] = TENSOR_SPEC(data->rms_f,         B * T * C);
    tensors[i++] = TENSOR_SPEC(data->rms_f_rstd,    B * T);
    tensors[i++] = TENSOR_SPEC(data->losses,        B * T);
    tensors[i++] = TENSOR_SPEC(data->output,        B * T * std::max(qkv_w, Vp));
    tensors[i++] = TENSOR_SPEC(data->scratch_btc,   B * T * C);
    tensors[i++] = TENSOR_SPEC(data->scratch_btc2,  B * T * C);
    tensors[i++] = TENSOR_SPEC(data->scratch_qkv3c, B * T * 3 * C);
    assert(i == NUM_ACTIVATION_TENSORS);
}

void *malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    printf0("allocating %d MiB for activations\n",
            (int)round(bytes / (1024.0 * 1024.0)));
    void *mem;
    cudaCheck(cudaMalloc(&mem, bytes));
    cudaCheck(cudaMemset(mem, 0, bytes));

    char *it = (char *)mem;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if (tensors[i].size == 0) {
            *(tensors[i].ptr) = nullptr;
        } else {
            *(tensors[i].ptr) = it;
            it += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return mem;
}

// ============================================================================
// SFNet model
// ============================================================================
typedef struct {
    SFNetConfig config;
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void *params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    ParameterTensors grads;
    void *grads_memory;
    float *m_memory;
    float *v_memory;
    float *master_weights;
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void *acts_memory;
    int batch_size;
    int seq_len;
    int *inputs;
    int *targets;
    float mean_loss;
    float *accumulated_mean_loss;
    float *cpu_losses;
    unsigned long long rng_state;
    unsigned long long rng_state_last_update;
    int use_master_weights;
    bool init_state;
    int *workload_indices;
    int4 *bucket_info;
    float2 *d_freqs_cis;
} SFNet;

// ============================================================================
// RoPE precomputation
// ============================================================================
static float2 *precompute_freqs_cis(int max_seq_len, int head_dim, float rope_theta) {
    int half = head_dim / 2;
    size_t n = (size_t)max_seq_len * half;
    float2 *h = (float2 *)mallocCheck(n * sizeof(float2));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            h[pos * half + i] = make_float2(cosf(angle), sinf(angle));
        }
    }
    float2 *d;
    cudaCheck(cudaMalloc(&d, n * sizeof(float2)));
    cudaCheck(cudaMemcpy(d, h, n * sizeof(float2), cudaMemcpyHostToDevice));
    free(h);
    return d;
}

// ============================================================================
// GQA expand / reduce
// ============================================================================
__global__ void expand_gqa_kernel(
    floatX *__restrict__ out,           // (BT, 3*NH*HD)
    const floatX *__restrict__ inp,     // (BT, (NH+2*NKV)*HD)
    int BT, int NH, int NKV, int n_rep, int HD)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int expanded_cols = 3 * NH * HD;
    int total = BT * expanded_cols;
    if (idx >= total) return;
    int compact_cols = (NH + 2 * NKV) * HD;
    int bt = idx / expanded_cols;
    int col = idx % expanded_cols;
    int d = col % HD;
    int h = col / HD;

    float v;
    size_t src_base = (size_t)bt * compact_cols;
    if (h < NH) {
        v = (float)inp[src_base + h * HD + d];
    } else if (h < 2 * NH) {
        int kv_h = (h - NH) / n_rep;
        v = (float)inp[src_base + NH * HD + kv_h * HD + d];
    } else {
        int kv_h = (h - 2 * NH) / n_rep;
        v = (float)inp[src_base + (NH + NKV) * HD + kv_h * HD + d];
    }
    out[idx] = (floatX)sfnet_q_fwd(v);
}

static void expand_gqa(floatX *out, const floatX *inp, int BT, int NH, int NKV,
                       int n_rep, int HD, cudaStream_t stream) {
    int total = BT * 3 * NH * HD;
    expand_gqa_kernel<<<CEIL_DIV(total, 256), 256, 0, stream>>>(
        out, inp, BT, NH, NKV, n_rep, HD);
    cudaCheck(cudaGetLastError());
}

__global__ void reduce_kv_grad_kernel(
    floatX *__restrict__ out,           // (BT, (NH+2*NKV)*HD)
    const floatX *__restrict__ inp,     // (BT, 3*NH*HD)
    int BT, int NH, int NKV, int n_rep, int HD)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t compact_stride = (size_t)(NH + 2 * NKV) * HD;
    const size_t expanded_stride = (size_t)3 * NH * HD;
    const size_t total = (size_t)BT * compact_stride;
    const size_t q_span = (size_t)NH * HD;
    const size_t k_span = (size_t)NKV * HD;

    for (; idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        size_t bt = idx / compact_stride;
        size_t o = idx - bt * compact_stride;
        size_t in_base = bt * expanded_stride;
        if (o < q_span) {
            out[idx] = inp[in_base + o];
        } else if (o < q_span + k_span) {
            size_t kk = o - q_span;
            int kv_h = (int)(kk / HD);
            int d = (int)(kk % HD);
            float acc = 0.0f;
            for (int r = 0; r < n_rep; r++) {
                int h = kv_h * n_rep + r;
                acc += (float)inp[in_base + q_span + (size_t)h * HD + d];
            }
            out[idx] = (floatX)acc;
        } else {
            size_t vv = o - (q_span + k_span);
            int kv_h = (int)(vv / HD);
            int d = (int)(vv % HD);
            float acc = 0.0f;
            for (int r = 0; r < n_rep; r++) {
                int h = kv_h * n_rep + r;
                acc += (float)inp[in_base + 2 * q_span + (size_t)h * HD + d];
            }
            out[idx] = (floatX)acc;
        }
    }
}

static void reduce_kv_grad(floatX *out, const floatX *inp, int BT, int NH,
                           int NKV, int n_rep, int HD, cudaStream_t stream) {
    const size_t compact_stride = (size_t)(NH + 2 * NKV) * HD;
    const size_t total = (size_t)BT * compact_stride;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    if (grid > 65535u) grid = 65535u;
    if (grid == 0u) grid = 1u;
    reduce_kv_grad_kernel<<<grid, block, 0, stream>>>(
        out, inp, BT, NH, NKV, n_rep, HD);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Init / checkpoint helpers
// ============================================================================
void sfnet_init_common(SFNet *model) {
    model->acts_memory = nullptr;
    model->inputs = nullptr;
    model->targets = nullptr;
    model->accumulated_mean_loss = nullptr;
    model->cpu_losses = nullptr;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    model->params_memory = nullptr;
    model->grads_memory = nullptr;
    model->workload_indices = nullptr;
    model->bucket_info = nullptr;
    model->m_memory = nullptr;
    model->v_memory = nullptr;
    model->master_weights = nullptr;
    model->d_freqs_cis = nullptr;
    model->rng_state = 13371337 + multi_gpu_config.process_rank;
#if defined(ENABLE_Q115)
    model->use_master_weights = 1;
#else
    model->use_master_weights = 0;
#endif
    model->init_state = true;
}

void sfnet_set_hyperparameters(SFNetConfig *cfg, const char *model_str) {
    cfg->dim = 768;
    cfg->n_layers = 12;
    cfg->n_heads = 12;
    cfg->n_kv_heads = 6;  // default for sfnet:c768 (NH=12/2); overwritten by sfnet:c parser
    cfg->ffn_dim = 2048;
    cfg->head_dim = 64;
    cfg->vocab_size = 50257;
    cfg->padded_vocab_size = 50304;
    cfg->max_seq_len = 1024;
    cfg->rope_theta = 10000.0f;
    cfg->norm_eps = 1e-3f;

    if (model_str && strncmp(model_str, "sfnet:c", 7) == 0) {
        int c = atoi(model_str + 7);
        if (c > 0) {
            cfg->dim = c;
            cfg->head_dim = 64;
            cfg->n_heads = c / 64;
            // qkv_w = (n_heads + 2*n_kv_heads)*head_dim must be a multiple of C
            // so that cuBLASLt can find a kernel for the (OC, B*T, C) GEMM.
            // The smallest such value with ≥1 KV head is 2*C (NKV = n_heads/2).
            cfg->n_kv_heads = cfg->n_heads / 2;  // qkv_w = 2*C, ratio 2:1 GQA

            // ffn_dim must also be a multiple of C for the up/gate/down GEMMs.
            // Round (8/3)*C up to the next multiple of C.
            {
                int ffn_raw = (8 * c) / 3;
                cfg->ffn_dim = ((ffn_raw + c - 1) / c) * c;
            }
        }
    }
    if (model_str) {
        const char *p = strstr(model_str, ":L");
        if (p) {
            int L = atoi(p + 2);
            if (L > 0) cfg->n_layers = L;
        }
    }
    cfg->alpha_init = 1.0f / sqrtf(2.0f * (float)cfg->n_layers);
}

void sfnet_allocate_weights(SFNet *model) {
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(
        &model->params, model->param_elements, model->param_sizeof);
}

void sfnet_allocate_state(SFNet *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n",
            (int)round(model->num_parameters * sizeof(floatX) / (1024.0 * 1024.0)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(
        &model->grads, model->param_elements, model->param_sizeof);

    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);

    cudaCheck(cudaMalloc((void **)&model->inputs,  B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&model->accumulated_mean_loss, sizeof(float)));
    cudaCheck(cudaMallocHost((void **)&model->cpu_losses, B * T * sizeof(float)));

    size_t num_c_groups = CEIL_DIV(model->config.dim, (WARP_SIZE * x128::size));
    model->workload_indices = (int *)mallocCheck(sizeof(int) * B * T * num_c_groups);
    model->bucket_info = (int4 *)mallocCheck(sizeof(int4) * B * T * num_c_groups);

    int memory_status = 0;
    size_t shard_np = multi_gpu_config.shard_num_parameters;
    memory_status |= cudaMallocConditionallyManaged((void **)&model->m_memory, shard_np * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void **)&model->v_memory, shard_np * sizeof(float));
    if (model->use_master_weights == 1) {
        memory_status |= cudaMallocConditionallyManaged(
            (void **)&model->master_weights, shard_np * sizeof(float));
    }
    if (memory_status >= 1) printf0("WARNING: Fell back to cudaMallocManaged\n");

    size_t free_b, total_b;
    cudaCheck(cudaMemGetInfo(&free_b, &total_b));
    printf0("device memory usage: %zd MiB / %zd MiB\n",
            (total_b - free_b) / 1024 / 1024, total_b / 1024 / 1024);
}

void sfnet_random_init(SFNet *model, const char *model_str) {
    sfnet_set_hyperparameters(&model->config, model_str);
    sfnet_allocate_weights(model);

    model->d_freqs_cis = precompute_freqs_cis(
        model->config.max_seq_len, model->config.head_dim, model->config.rope_theta);

    float init_scale = 0.02f;
#if defined(ENABLE_Q115)
    init_scale = 0.1f;
#endif
    mt19937_state rng;
    manual_seed(&rng, 42);

    floatX *cpu = (floatX *)mallocCheck(model->num_parameters_bytes);
    memset(cpu, 0, model->num_parameters_bytes);

    size_t L = (size_t)model->config.n_layers;
    float residual_scale = 1.0f / sqrtf(2.0f * (float)L);

    size_t offsets[NUM_PARAMETER_TENSORS + 1];
    offsets[0] = 0;
    for (int t = 0; t < NUM_PARAMETER_TENSORS; t++)
        offsets[t + 1] = offsets[t] + model->param_elements[t];

    for (int t = 0; t < NUM_PARAMETER_TENSORS; t++) {
        size_t n = model->param_elements[t];
        if (t == 1 || t == 3 || t == 4 || t == 10) {
            // RMSNorm / per-head RMSNorm weights — init to 1.0
            for (size_t j = 0; j < n; j++) cpu[offsets[t] + j] = (floatX)1.0f;
            continue;
        }
        if (t == 9) {
            // alpha — per-layer residual scale
            for (size_t j = 0; j < n; j++) cpu[offsets[t] + j] = (floatX)model->config.alpha_init;
            continue;
        }
        // Linear-projection weights
        float scale = init_scale;
        if (t == 5 || t == 8) scale *= residual_scale;  // attn_ow, down_w
        float *buf = (float *)mallocCheck(n * sizeof(float));
        normal_(buf, n, 0.0f, scale, &rng);
        for (size_t j = 0; j < n; j++) {
            float v = buf[j];
#if defined(ENABLE_Q115)
            v = fmaxf(-0.999969482421875f, fminf(0.999969482421875f, v));
#endif
            cpu[offsets[t] + j] = (floatX)v;
        }
        free(buf);
    }
    cudaCheck(cudaMemcpy(model->params_memory, cpu, model->num_parameters_bytes,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    free(cpu);
    printf0("SFNet: %zu parameters (%.2f M)\n",
            model->num_parameters, (double)model->num_parameters / 1e6);
}

void sfnet_write_checkpoint(SFNet *model, const char *path) {
    printf0("Writing checkpoint to %s\n", path);
    FILE *f = fopenCheck(path, "wb");
    int header[256];
    memset(header, 0, sizeof(header));
    header[0]  = 20260519; // SFNet magic
    header[1]  = PRECISION_MODE;
    header[2]  = model->config.dim;
    header[3]  = model->config.n_layers;
    header[4]  = model->config.n_heads;
    header[5]  = model->config.n_kv_heads;
    header[6]  = model->config.ffn_dim;
    header[7]  = model->config.head_dim;
    header[8]  = model->config.vocab_size;
    header[9]  = model->config.padded_vocab_size;
    header[10] = model->config.max_seq_len;
    fwriteCheck(header, sizeof(int), 256, f);
    device_to_file(f, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    fcloseCheck(f);
}

ShardInfo sfnet_get_tensor_at_layer(const SFNet *model, int layer_id, int tensor_id) {
    ptrdiff_t offset = 0;
    for (int i = 0; i < tensor_id; i++) offset += (ptrdiff_t)model->param_elements[i];
    size_t size = model->param_elements[tensor_id];
    if (tensor_id >= 1 && tensor_id <= 9) {
        size /= model->config.n_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

// ============================================================================
// Pull per-layer α from device into host vector
// ============================================================================
static void pull_alpha(const SFNet *model, std::vector<float> &h_alpha) {
    size_t L = (size_t)model->config.n_layers;
    h_alpha.resize(L);
    std::vector<floatX> tmp(L);
    cudaCheck(cudaMemcpy(tmp.data(), model->params.alpha, L * sizeof(floatX),
                         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < L; i++) h_alpha[i] = (float)tmp[i];
}

// ============================================================================
// Forward pass
// ============================================================================
void sfnet_forward(SFNet *model, const int *inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    if (model->params_memory == nullptr) {
        fprintf(stderr, "Error: model not initialized.\n"); exit(EXIT_FAILURE);
    }
    if ((int)B > model->batch_size || (int)T > model->seq_len) {
        fprintf(stderr, "Model: B=%d T=%d, Desired: B=%d T=%d\n",
                model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }
    const size_t C   = model->config.dim;
    const size_t L   = model->config.n_layers;
    const size_t NH  = model->config.n_heads;
    const size_t NKV = model->config.n_kv_heads;
    const size_t HD  = model->config.head_dim;
    const size_t FFN = model->config.ffn_dim;
    const size_t Vp  = model->config.padded_vocab_size;
    const int V      = model->config.vocab_size;
    const int n_rep  = (int)(NH / NKV);
    const size_t qkv_w = (NH + 2 * NKV) * HD;

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(inputs, B * T, V);

    ParameterTensors params = model->params;
    ActivationTensors acts  = model->acts;
    const float eps = model->config.norm_eps;

    std::vector<float> h_alpha;
    pull_alpha(model, h_alpha);

    // 1. Token embedding (no positional table — RoPE inside attention handles position)
    encoder_forward(acts.encoded, model->inputs, params.wte, /*wpe=*/nullptr,
                    B, T, C, main_stream);

    // 2. Transformer blocks
    for (int l = 0; l < (int)L; l++) {
        NvtxRange layer_range("Layer", l);
        floatX *x      = (l == 0) ? acts.encoded
                                    : acts.residual + (size_t)(l - 1) * B * T * C;
        floatX *x_next = acts.residual + (size_t)l * B * T * C;

        floatX *l_rms1w   = params.rms1w    + (size_t)l * C;
        floatX *l_qkvw    = params.qkvw     + (size_t)l * qkv_w * C;
        floatX *l_qnw     = params.q_norm_w + (size_t)l * NH * HD;
        floatX *l_knw     = params.k_norm_w + (size_t)l * NKV * HD;
        floatX *l_attn_ow = params.attn_ow  + (size_t)l * C * C;
        floatX *l_gate_w  = params.gate_w   + (size_t)l * FFN * C;
        floatX *l_up_w    = params.up_w     + (size_t)l * FFN * C;
        floatX *l_down_w  = params.down_w   + (size_t)l * C * FFN;

        floatX *l_rms1     = acts.rms1      + (size_t)l * B * T * C;
        float  *l_rms1r    = acts.rms1_rstd + (size_t)l * B * T;
        floatX *l_qkv_pre  = acts.qkv_pre   + (size_t)l * B * T * qkv_w;
        floatX *l_qkv_post = acts.qkv_post  + (size_t)l * B * T * qkv_w;
        float  *l_rstd_q   = acts.rstd_q    + (size_t)l * B * T * NH;
        float  *l_rstd_k   = acts.rstd_k    + (size_t)l * B * T * NKV;
        floatX *l_qkvr_perm= acts.qkvr_perm + (size_t)l * B * T * 3 * C;
        floatX *l_atty     = acts.atty      + (size_t)l * B * T * C;
        floatX *l_gate     = acts.gate      + (size_t)l * B * T * FFN;
        floatX *l_up       = acts.up        + (size_t)l * B * T * FFN;
        floatX *l_glu      = acts.glu       + (size_t)l * B * T * FFN;

        float alpha = h_alpha[l];

        // 2a. Pre-block RMSNorm
        rmsnorm_forward(l_rms1, l_rms1r, x, l_rms1w, B, T, C, eps, main_stream);

        // 2b. Fused QKV projection
        sfnet_matmul_forward(l_qkv_pre, l_rms1, l_qkvw, B, T, C, (int)qkv_w, main_stream);

        // 2c. Copy qkv_pre -> qkv_post so we can do in-place QK-Norm+RoPE on
        //     qkv_post while preserving qkv_pre for backward.
        cudaCheck(cudaMemcpyAsync(l_qkv_post, l_qkv_pre,
                                  B * T * qkv_w * sizeof(floatX),
                                  cudaMemcpyDeviceToDevice, main_stream));

        // 2d. QK-Norm: per-head RMSNorm over HD on Q and K slices (V untouched)
        qhead_rmsnorm_forward(l_qkv_post, l_rstd_q, l_qnw,
                              B * T, (int)qkv_w, /*slice_offset=*/0,
                              (int)NH, (int)HD, eps, main_stream);
        qhead_rmsnorm_forward(l_qkv_post, l_rstd_k, l_knw,
                              B * T, (int)qkv_w, /*slice_offset=*/(int)(NH * HD),
                              (int)NKV, (int)HD, eps, main_stream);

        // 2e. RoPE on Q,K
        rope_qk_forward(l_qkv_post, model->d_freqs_cis, B, T, NH, NKV, HD, main_stream);

        // 2f. Expand GQA → (B,T,3*NH*HD) into scratch_qkv3c (input to attention)
        expand_gqa(acts.scratch_qkv3c, l_qkv_post, B * T, NH, NKV, n_rep, HD, main_stream);

        // 2g. Attention (flash, SF16-aware softmax baked in)
        //   attention_forward(out, qkvr_output, att_scratch, inp, B, T, C, NH, stream)
        attention_forward(l_atty, l_qkvr_perm, acts.att, acts.scratch_qkv3c,
                          B, T, C, NH, main_stream);

        // 2h. Attention output projection
        sfnet_matmul_forward(acts.attn_out, l_atty, l_attn_ow, B, T, C, C, main_stream);

        // 2i. MLP gate & up projections (parallel branch — same x_norm)
        sfnet_matmul_forward(l_gate, l_rms1, l_gate_w, B, T, C, (int)FFN, main_stream);
        sfnet_matmul_forward(l_up,   l_rms1, l_up_w,   B, T, C, (int)FFN, main_stream);

        // 2j. Tanh-GLU
        tanh_glu_forward(l_glu, l_gate, l_up, B * T * FFN, main_stream);

        // 2k. MLP down projection
        sfnet_matmul_forward(acts.mlp_out, l_glu, l_down_w, B, T, (int)FFN, C, main_stream);

        // 2l. Single norm-preserving scaled residual
        scaled_residual_3way_forward(x_next, x, acts.attn_out, acts.mlp_out,
                                     alpha, B * T * C, main_stream);
    }

    // 3. Final RMSNorm
    floatX *x_final = acts.residual + (size_t)(L - 1) * B * T * C;
    rmsnorm_forward(acts.rms_f, acts.rms_f_rstd, x_final, params.rms_fw,
                    B, T, C, eps, main_stream);

    // 4. Tied LM head — is_logits=true so Q1.15 forward-clamp is NOT applied
    // to logits (clamping would saturate softmax and lock loss at ln(V)).
    sfnet_matmul_forward(acts.output, acts.rms_f, params.wte,
                         B, T, C, Vp, main_stream, /*is_logits=*/true);
}

// ============================================================================
// Validation
// ============================================================================
float sfnet_validate(SFNet *model, const int *inputs, const int *targets,
                     size_t B, size_t T) {
    assert(targets != nullptr);
    sfnet_forward(model, inputs, B, T);
    const size_t Vp = model->config.padded_vocab_size;
    const int V = model->config.vocab_size;
    ActivationTensors acts = model->acts;
    const float dloss = 1.0f / (float)(B * T);
    cudaCheck(cudaMemset(acts.losses, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B * T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets,
                     B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float mean_loss = 0.0f;
    for (int i = 0; i < (int)(B * T); i++) mean_loss += model->cpu_losses[i];
    mean_loss /= (float)(B * T);
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

// ============================================================================
// Backward pass
// ============================================================================
void sfnet_backward_and_reduce(SFNet *model, int *inputs, const int *targets,
                                int grad_accum_steps, int micro_step) {
    if (model->grads_memory == nullptr) {
        fprintf(stderr, "Allocate gradients before backward\n"); exit(EXIT_FAILURE);
    }
    NVTX_RANGE_FN();
    bool last_step = (micro_step == grad_accum_steps - 1);

    if (micro_step == 0) {
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0,
            model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0,
            model->num_parameters * sizeof(floatX), main_stream));
    }

    const size_t B   = model->batch_size;
    const size_t T   = model->seq_len;
    const size_t C   = model->config.dim;
    const size_t L   = model->config.n_layers;
    const size_t NH  = model->config.n_heads;
    const size_t NKV = model->config.n_kv_heads;
    const size_t HD  = model->config.head_dim;
    const size_t FFN = model->config.ffn_dim;
    const size_t Vp  = model->config.padded_vocab_size;
    const int V      = model->config.vocab_size;
    const int n_rep  = (int)(NH / NKV);
    const size_t qkv_w = (NH + 2 * NKV) * HD;

    ParameterTensors params = model->params;
    ParameterTensors grads  = model->grads;
    ActivationTensors acts  = model->acts;

    const float dloss = 1.0f / (float)(B * T * grad_accum_steps);
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B * T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets,
                     B, T, V, Vp, True, main_stream);

    // 1. Backward through LM head (tied: accumulates into grads.wte)
    //    scratch_btc  → dresidual (residual-stream gradient accumulator)
    //    scratch_btc2 → temporary d(rms_f), then later repurposed as d(rms1)
    floatX *dresidual = (floatX *)acts.scratch_btc;
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    floatX *dl_rmsf = (floatX *)acts.scratch_btc2;      // (B,T,C)
    sfnet_matmul_backward(dl_rmsf, grads.wte, acts.output, acts.rms_f, params.wte,
                          B, T, C, Vp, false, main_stream);

    // 2. Backward through final RMSNorm
    floatX *x_final = acts.residual + (size_t)(L - 1) * B * T * C;
    float *scratchF = (float *)acts.output;             // reuse acts.output as float scratch
    rmsnorm_backward(dresidual, grads.rms_fw, scratchF,
                     dl_rmsf, x_final, params.rms_fw, acts.rms_f_rstd,
                     B, T, C, main_stream);

    // From here on, scratch_btc2 is repurposed as dl_rms1 (per-layer d(rms1) accumulator).
    floatX *dl_rms1 = (floatX *)acts.scratch_btc2;

    std::vector<float> h_alpha;
    pull_alpha(model, h_alpha);

    // ---- Backward through transformer blocks ----
    for (int l = (int)L - 1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        floatX *x_in = (l == 0) ? acts.encoded
                                  : acts.residual + (size_t)(l - 1) * B * T * C;

        // Parameter & grad pointers
        floatX *l_rms1w   = params.rms1w    + (size_t)l * C;
        floatX *l_qkvw    = params.qkvw     + (size_t)l * qkv_w * C;
        floatX *l_qnw     = params.q_norm_w + (size_t)l * NH * HD;
        floatX *l_knw     = params.k_norm_w + (size_t)l * NKV * HD;
        floatX *l_attn_ow = params.attn_ow  + (size_t)l * C * C;
        floatX *l_gate_w  = params.gate_w   + (size_t)l * FFN * C;
        floatX *l_up_w    = params.up_w     + (size_t)l * FFN * C;
        floatX *l_down_w  = params.down_w   + (size_t)l * C * FFN;

        floatX *dl_rms1w   = grads.rms1w    + (size_t)l * C;
        floatX *dl_qkvw    = grads.qkvw     + (size_t)l * qkv_w * C;
        floatX *dl_qnw     = grads.q_norm_w + (size_t)l * NH * HD;
        floatX *dl_knw     = grads.k_norm_w + (size_t)l * NKV * HD;
        floatX *dl_attn_ow = grads.attn_ow  + (size_t)l * C * C;
        floatX *dl_gate_w  = grads.gate_w   + (size_t)l * FFN * C;
        floatX *dl_up_w    = grads.up_w     + (size_t)l * FFN * C;
        floatX *dl_down_w  = grads.down_w   + (size_t)l * C * FFN;

        // Activation pointers
        floatX *l_rms1     = acts.rms1      + (size_t)l * B * T * C;
        float  *l_rms1r    = acts.rms1_rstd + (size_t)l * B * T;
        floatX *l_qkv_pre  = acts.qkv_pre   + (size_t)l * B * T * qkv_w;
        floatX *l_qkv_post = acts.qkv_post  + (size_t)l * B * T * qkv_w;
        float  *l_rstd_q   = acts.rstd_q    + (size_t)l * B * T * NH;
        float  *l_rstd_k   = acts.rstd_k    + (size_t)l * B * T * NKV;
        floatX *l_qkvr_perm= acts.qkvr_perm + (size_t)l * B * T * 3 * C;
        floatX *l_atty     = acts.atty      + (size_t)l * B * T * C;
        floatX *l_gate     = acts.gate      + (size_t)l * B * T * FFN;
        floatX *l_up       = acts.up        + (size_t)l * B * T * FFN;
        floatX *l_glu      = acts.glu       + (size_t)l * B * T * FFN;

        float alpha = h_alpha[l];

        // ---- Scratch allocation inside acts.output (size B*T*max(qkv_w, Vp)) ----
        // FFN > C and 3*C >= qkv_w for typical configs.  We need:
        //   d_glu, d_gate, d_up         : 3 × B*T*FFN
        //   d_atty                       :     B*T*C
        //   d_qkvfull                    :     B*T*3*C
        //   d_qkv_compact                :     B*T*qkv_w
        // We never need all of these simultaneously (sequential layer backward
        // through MLP branch -> attention branch).  We sub-divide acts.output as:
        floatX *out_base = (floatX *)acts.output;
        floatX *d_glu  = out_base;
        floatX *d_gate = out_base + (size_t)B * T * FFN;
        floatX *d_up   = out_base + (size_t)2 * B * T * FFN;
        // After MLP backward finishes we reuse the same memory region:
        floatX *d_atty       = out_base;
        floatX *d_qkvfull    = out_base;                                  // (B,T,3*NH*HD)
        floatX *d_qkv_compact = out_base + (size_t)B * T * 3 * C;          // (B,T,qkv_w)

        // ---- Backward scaled residual ----
        // We obtain (d_attn_out, d_mlp_out, dx_in) from dresidual.
        // attn_out and mlp_out activations are stored in acts.attn_out / acts.mlp_out
        // but those are SCRATCH (overwritten every forward).  Their values are NOT
        // saved per-layer; however we don't need them: the gradient of the scaled
        // residual w.r.t. each branch is just c2*dresidual and the gradient w.r.t.
        // x is c1*dresidual.  We treat α as a hyperparameter (no α-gradient).
        floatX *d_attn_branch = (floatX *)acts.attn_out;  // reuse as gradient buffer
        floatX *d_mlp_branch  = (floatX *)acts.mlp_out;
        scaled_residual_3way_backward(dresidual, d_attn_branch, d_mlp_branch,
                                      dresidual, alpha, B * T * C,
                                      /*accumulate_dx=*/false, main_stream);

        // ---- Backward MLP branch ----
        // d(down):   d_glu = d_mlp_branch @ down_w  ;  dl_down_w += d_mlp_branch^T @ glu
        sfnet_matmul_backward(d_glu, dl_down_w, d_mlp_branch, l_glu, l_down_w,
                              B, T, FFN, C, false, main_stream);
        // tanh-GLU backward → d_gate, d_up
        tanh_glu_backward(d_gate, d_up, d_glu, l_gate, l_up, B * T * FFN, main_stream);

        // d(gate proj): dl_rms1  = d_gate @ gate_w ; dl_gate_w += d_gate^T @ rms1
        sfnet_matmul_backward(dl_rms1, dl_gate_w, d_gate, l_rms1, l_gate_w,
                              B, T, C, FFN, false, main_stream);
        // d(up   proj): dl_rms1 += d_up   @ up_w   ; dl_up_w   += d_up^T   @ rms1
        sfnet_matmul_backward(dl_rms1, dl_up_w,   d_up,   l_rms1, l_up_w,
                              B, T, C, FFN, true,  main_stream);

        // ---- Backward attention branch ----
        // d(attn output proj): d_atty = d_attn_branch @ attn_ow ; dl_attn_ow += d_attn_branch^T @ atty
        sfnet_matmul_backward(d_atty, dl_attn_ow, d_attn_branch, l_atty, l_attn_ow,
                              B, T, C, C, false, main_stream);

        // attention backward:
        //   reads qkvr (permuted Q,K,V) saved during forward
        //   recomputes att internally, then computes dq,dk,dv into dqkvr (permuted)
        //   re-permutes and writes dinp = grad w.r.t. expanded (B,T,3*NH*HD) input
        // We need disjoint scratch buffers for `datt` and `att`, plus a
        // (B,T,C) scratch.  Strategy:
        //   dqkvr      → acts.scratch_qkv3c       (3*B*T*C)
        //   att        → acts.att                 (B*NH*T*T)  -- written by softmax
        //   datt       → carve out from acts.output at offset (B*T*C + B*T*3*C)
        //                so it does NOT collide with d_atty (offset 0) or
        //                d_qkvfull (offset B*T*C, size B*T*3*C)
        //   scratch    → reuse l_atty (already consumed by attn_ow backward)
        floatX *dqkvr_scr  = acts.scratch_qkv3c;
        floatX *datt_scr   = out_base + (size_t)B * T * (C + 3 * C);
        floatX *attn_scr_btc = l_atty;
        attention_backward(d_qkvfull, dqkvr_scr, datt_scr, attn_scr_btc,
                           d_atty, l_qkvr_perm, acts.att,
                           B, T, C, NH, main_stream);

        // d_qkvfull is now in expanded (B,T,3*NH*HD) layout.
        // Reduce to compact (B,T,(NH+2*NKV)*HD).
        reduce_kv_grad(d_qkv_compact, d_qkvfull, B * T, NH, NKV, n_rep, HD, main_stream);

        // Un-rotate Q,K in compact form (in place)
        rope_qk_backward(d_qkv_compact, model->d_freqs_cis, B, T, NH, NKV, HD, main_stream);

        // QK-Norm backward (K first, then Q — they touch disjoint slices)
        qhead_rmsnorm_backward(d_qkv_compact, dl_knw,
                               l_qkv_pre, l_knw, l_rstd_k,
                               B * T, (int)qkv_w, /*slice_offset=*/(int)(NH * HD),
                               (int)NKV, (int)HD, main_stream);
        qhead_rmsnorm_backward(d_qkv_compact, dl_qnw,
                               l_qkv_pre, l_qnw, l_rstd_q,
                               B * T, (int)qkv_w, /*slice_offset=*/0,
                               (int)NH, (int)HD, main_stream);

        // QKV proj backward:  dl_rms1 += d_qkv_compact @ qkvw  ;  dl_qkvw += d_qkv_compact^T @ rms1
        sfnet_matmul_backward(dl_rms1, dl_qkvw, d_qkv_compact, l_rms1, l_qkvw,
                              B, T, C, (int)qkv_w, true, main_stream);

        // ---- Backward pre-block RMSNorm ----
        rmsnorm_backward(dresidual, dl_rms1w, scratchF,
                         dl_rms1, x_in, l_rms1w, l_rms1r, B, T, C, main_stream);

        if (last_step) {
            floatX *const ptrs[] = {dl_rms1w, dl_qkvw, dl_qnw, dl_knw,
                                    dl_attn_ow, dl_gate_w, dl_up_w, dl_down_w};
            const size_t nelems[] = {C, qkv_w * C, NH * HD, NKV * HD,
                                     C * C, FFN * C, FFN * C, C * FFN};
            multi_gpu_async_reduce_gradient(ptrs, nelems, &multi_gpu_config, main_stream);
        }
    }

    // ---- Backward through token embedding ----
    encoder_backward(grads.wte, nullptr, (floatX *)acts.output,
                     model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C,
                     random_u32(&model->rng_state), main_stream);

    if (last_step) {
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B * T, main_stream);
#if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss,
                                model->accumulated_mean_loss, sizeof(float),
                                ncclFloat, ncclAvg, multi_gpu_config.nccl_comm,
                                main_stream));
#endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss,
                                  sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        floatX *const ptrs[] = {grads.wte, grads.rms_fw};
        const size_t nelems[] = {(size_t)Vp * C, C};
        multi_gpu_async_reduce_gradient(ptrs, nelems, &multi_gpu_config, main_stream);
    }
    cudaCheck(cudaStreamSynchronize(main_stream));
    if (last_step) model->mean_loss /= (float)(B * T * grad_accum_steps);
    else           model->mean_loss = -1.0f;
}

// ============================================================================
// Gradient norm / sanitize
// ============================================================================
__global__ void sanitize_nonfinite_kernel(floatX *grad, size_t n, unsigned int *bad_count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (size_t)blockDim.x * gridDim.x) {
        float g = (float)grad[idx];
        if (!isfinite(g) || fabsf(g) > 1.0e6f) {
            grad[idx] = (floatX)0;
            atomicAdd(bad_count, 1u);
        }
    }
}

float sfnet_calculate_grad_norm(SFNet *model, MultiGpuConfig *mgc) {
    NVTX_RANGE_FN();
    floatX *gm = (floatX *)model->grads_memory;
    float *gns = (float *)model->acts.output;
    unsigned int *bad_count = (unsigned int *)gns;

    cudaCheck(cudaMemsetAsync(bad_count, 0, sizeof(unsigned int), main_stream));
    const int block = 256;
    unsigned int grid = (unsigned int)((model->num_parameters + block - 1) / block);
    if (grid > 65535u) grid = 65535u;
    if (grid == 0u) grid = 1u;
    sanitize_nonfinite_kernel<<<grid, block, 0, main_stream>>>(
        gm, model->num_parameters, bad_count);
    cudaCheck(cudaGetLastError());

    unsigned int bad_cpu = 0;
    cudaCheck(cudaMemcpy(&bad_cpu, bad_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    g_last_sanitized_grad_count = bad_cpu;

    int num_slices[2] = {1, model->config.n_layers};
    int max_sums = get_max_num_block_sums(num_slices, 2);
    global_norm_squared(gns, gm, model->num_parameters, 0, 1, max_sums, true, main_stream);
    global_sum_deterministic(gns, gns, max_sums, main_stream);
    float gns_cpu = 0.0f;
    cudaCheck(cudaMemcpy(&gns_cpu, gns, sizeof(float), cudaMemcpyDeviceToHost));
    return sqrtf(gns_cpu);
}

// ============================================================================
// AdamW update
// ============================================================================
void sfnet_update(SFNet *model, float lr, float beta1, float beta2,
                  float eps, float weight_decay, float grad_scale, int t,
                  MultiGpuConfig *mgc, bool init_from_master_only = false) {
    NVTX_RANGE_FN();
    if (!model->grads_memory || !model->m_memory || !model->v_memory) {
        fprintf(stderr, "Allocate optimizer state before update\n"); exit(EXIT_FAILURE);
    }
    bool init_state = model->init_state;
    if (init_state) {
        model->init_state = false;
        cudaCheck(cudaMemset(model->m_memory, 0, mgc->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, mgc->shard_num_parameters * sizeof(float)));
    }
    model->rng_state_last_update = model->rng_state;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        unsigned int seed = random_u32(&model->rng_state);
        int num_layers = model->config.n_layers;
        if (i == 0 || i == 10) num_layers = 1;  // non-per-layer tensors

        ShardInfo tensor = sfnet_get_tensor_at_layer(model, 0, i);
        ShardInfo shard  = multi_gpu_get_shard_offset(tensor.size, mgc, 1);
        ptrdiff_t lof = tensor.offset + shard.offset;
        ptrdiff_t lop = tensor.offset / mgc->num_processes;

#if defined(ENABLE_Q115)
        // Q1.15: only decay down_w (8)
        float wd = (i == 8) ? weight_decay : 0.0f;
#else
        float wd = (i == 0 || i == 2 || i == 5 || i == 6 || i == 7 || i == 8)
                   ? weight_decay : 0.0f;
#endif

        floatX *param_ptr = (floatX *)model->params_memory + lof;
        floatX *grad_ptr  = (floatX *)model->grads_memory  + lof;
        ptrdiff_t opt_off = (mgc->zero_stage < 1) ? lof : lop;
        float *m_ptr      = model->m_memory + opt_off;
        float *v_ptr      = model->v_memory + opt_off;
        float *master_ptr = model->master_weights ? model->master_weights + opt_off : nullptr;

        if (init_state && master_ptr) {
            size_t gs = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(gs, num_layers), 512, 0, main_stream>>>(
                master_ptr, param_ptr, shard.size, shard.size, tensor.size);
            cudaCheck(cudaGetLastError());
        }
        if (init_from_master_only) {
            init_from_master(param_ptr, master_ptr, shard.size, tensor.size,
                             shard.size, num_layers, seed, main_stream);
        } else {
            adamw_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr,
                         shard.size, tensor.size, tensor.size, shard.size,
                         num_layers, lr, beta1, beta2, t, eps, wd,
                         grad_scale, seed, main_stream);
        }
    }
    cudaCheck(cudaStreamSynchronize(main_stream));
}

// ============================================================================
// MFU estimate
// ============================================================================
float sfnet_estimate_mfu(SFNet *model, int num_tokens, float dt) {
    float N = (float)model->num_parameters;
    float flops = 6.0f * N * (float)num_tokens;
    float promised = get_flops_promised(deviceProp.name, PRECISION_MODE);
    if (dt <= 0.0f || promised <= 0.0f) return 0.0f;
    return flops / (dt * promised * 1e12f);
}

// ============================================================================
// main()
// ============================================================================
int main(int argc, char *argv[]) {
    const char *model_str    = "sfnet:c768";
    // Default to the FineWeb 10B subset for pretraining (see dev/data/fineweb.py).
    // Run `python dev/data/fineweb.py --version 10B` to materialize these shards.
    const char *input_bin    = "dev/data/fineweb10B/fineweb_train_*.bin";
    const char *input_val_bin= "dev/data/fineweb10B/fineweb_val_*.bin";
    const char *output_dir   = "";
    int batch_size           = 4;
    int sequence_length      = 1024;
    int total_batch_size     = 0;
    int num_iterations       = 50;
    int inference_only       = 0;  // currently parsed but not used; reserved
    float learning_rate      = 3e-4f;
    int warmup_iters         = 0;
    float grad_clip          = 1.0f;
    float lr_decay_frac      = 1.0f;
    float weight_decay       = 0.0f;
    int val_loss_every       = 10;
    int val_max_steps        = 5;
    int overfit_single_batch = 0;
    int tensorcores          = 1;
    const char *device_str   = "";
    int zero_stage           = 0;
    const char *dtype_str    = "bfloat16";
    const char *tokenizer_bin = "gpt2_tokenizer.bin";

    for (int i = 1; i < argc; i++) {
#define PARSE_STR(flag, var) if (strcmp(argv[i], flag) == 0) { var = argv[++i]; continue; }
#define PARSE_INT(flag, var) if (strcmp(argv[i], flag) == 0) { var = atoi(argv[++i]); continue; }
#define PARSE_FLT(flag, var) if (strcmp(argv[i], flag) == 0) { var = atof(argv[++i]); continue; }
        PARSE_STR("-e", model_str)
        PARSE_STR("--model", model_str)
        PARSE_STR("-i", input_bin)
        PARSE_STR("--input_bin", input_bin)
        PARSE_STR("-j", input_val_bin)
        PARSE_STR("--input_val_bin", input_val_bin)
        PARSE_STR("-o", output_dir)
        PARSE_STR("--output_dir", output_dir)
        PARSE_INT("-b", batch_size)
        PARSE_INT("--batch_size", batch_size)
        PARSE_INT("-t", sequence_length)
        PARSE_INT("--sequence_length", sequence_length)
        PARSE_INT("-d", total_batch_size)
        PARSE_INT("--total_batch_size", total_batch_size)
        PARSE_INT("-x", num_iterations)
        PARSE_INT("--num_iterations", num_iterations)
        PARSE_INT("--inference_only", inference_only)
        PARSE_FLT("-l", learning_rate)
        PARSE_FLT("--learning_rate", learning_rate)
        PARSE_INT("-u", warmup_iters)
        PARSE_INT("--warmup_iters", warmup_iters)
        PARSE_FLT("-q", lr_decay_frac)
        PARSE_FLT("-c", weight_decay)
        PARSE_FLT("--weight_decay", weight_decay)
        PARSE_FLT("--grad_clip", grad_clip)
        PARSE_INT("-v", val_loss_every)
        PARSE_INT("--val_loss_every", val_loss_every)
        PARSE_INT("-w", val_max_steps)
        PARSE_INT("--val_max_steps", val_max_steps)
        PARSE_INT("--overfit_single_batch", overfit_single_batch)
        PARSE_INT("--tensorcores", tensorcores)
        PARSE_STR("--device", device_str)
        PARSE_INT("-z", zero_stage)
        PARSE_INT("--zero_stage", zero_stage)
        PARSE_STR("--dtype", dtype_str)
        PARSE_STR("--tokenizer_bin", tokenizer_bin)
        fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(EXIT_FAILURE);
    }

    Tokenizer tokenizer = {};
    if (tokenizer_bin != nullptr && strlen(tokenizer_bin) > 0) {
        tokenizer_init(&tokenizer, tokenizer_bin);
        if (tokenizer.init_ok) {
            printf("Loaded tokenizer: vocab=%u eot=%d\n",
                   tokenizer.vocab_size, tokenizer.eot_token);
        }
    }

    multi_gpu_config = multi_gpu_config_init(-1, -1, -1, (char *)"", (char *)"", (char *)"");
    bool master_process = (multi_gpu_config.process_rank == 0);
    int ddp_rank = multi_gpu_config.process_rank;
    int ddp_world_size = multi_gpu_config.num_processes;
    int ddp_local_rank = multi_gpu_config.local_device_idx;

    (void)device_str;
    int device_id = ddp_local_rank;
    cudaCheck(cudaSetDevice(device_id));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, device_id));
    cudaCheck(cudaStreamCreate(&main_stream));

    printf0("Device: %s | SM %d.%d\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf0("Using model: %s | dtype: %s\n", model_str, dtype_str);
#if defined(ENABLE_Q115)
    printf0("SF16/Q1.15 mode enabled\n");
#endif

    (void)inference_only;
    (void)tensorcores;  // SFNet always uses tensor cores via cuBLASLt; flag kept for CLI compat
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // Match train_gpt2: CUBLAS_COMPUTE_32F gives the widest algorithm coverage
    // for cuBLASLt's heuristic with BF16 I/O.  CUBLAS_COMPUTE_32F_FAST_16BF
    // restricts the search to a specific BF16 tensor-core kernel family which
    // has poor coverage for non-canonical OC widths (e.g. GQA's 1536, 1280).
    bool enable_tf32 = (PRECISION_MODE == PRECISION_FP32) && deviceProp.major >= 8;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    int B = batch_size;
    int T = sequence_length;
    assert(T >= 1 && T <= 8192);
    if (total_batch_size == 0) total_batch_size = B * T * ddp_world_size;
    int tokens_per_fwdbwd = B * T * ddp_world_size;
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    printf0("total batch size: %d | grad_accum_steps: %d\n",
            total_batch_size, grad_accum_steps);

    char *logfile = nullptr;
    if (output_dir && strlen(output_dir) > 0) {
        create_dir_if_not_exists(output_dir);
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/main.log", output_dir);
        logfile = filename_buffer;
        FILE *lf = fopen(logfile, "w"); if (lf) fclose(lf);
    }

    SFNet model;
    sfnet_init_common(&model);
    sfnet_random_init(&model, model_str);
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);

    DataLoader train_loader;
    dataloader_init(&train_loader, input_bin, B, T, ddp_rank, ddp_world_size, 1);
    if (num_iterations == -1) {
        num_iterations = train_loader.num_tokens / total_batch_size;
    }
    printf0("Training data: %lld tokens, %d steps\n",
            (long long)train_loader.num_tokens, num_iterations);

    DataLoader val_loader;
    bool has_val = (strlen(input_val_bin) > 0);
    if (has_val) dataloader_init(&val_loader, input_val_bin, B, T, ddp_rank, ddp_world_size, 0);

    sfnet_allocate_state(&model, B, T);

    LearningRateScheduler lr_sched;
    lr_scheduler_init(&lr_sched, "cosine", learning_rate, warmup_iters,
                      num_iterations, lr_decay_frac);

    OutlierDetector loss_detector, grad_norm_detector;
    init_detector(&loss_detector);
    init_detector(&grad_norm_detector);

    cudaEvent_t ev_start, ev_end;
    cudaCheck(cudaEventCreate(&ev_start));
    cudaCheck(cudaEventCreate(&ev_end));

    float norm = -1.0f;
    for (int step = 0; step <= num_iterations; step++) {
        bool last_step = (step == num_iterations);

        if (has_val && val_loss_every > 0 && (step % val_loss_every == 0 || last_step)) {
            model.mean_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int s = 0; s < val_max_steps; s++) {
                dataloader_next_batch(&val_loader);
                model.mean_loss += sfnet_validate(&model, val_loader.inputs,
                                                   val_loader.targets, B, T);
            }
            model.mean_loss /= val_max_steps;
            printf0("val loss %f | val perplexity %f\n",
                    model.mean_loss, expf(model.mean_loss));
            if (master_process && logfile) {
                FILE *lf = fopen(logfile, "a");
                if (lf) { fprintf(lf, "s:%d tel:%f\n", step, model.mean_loss); fclose(lf); }
            }
        }

        if (last_step) break;

        cudaCheck(cudaEventRecord(ev_start));
        for (int micro = 0; micro < grad_accum_steps; micro++) {
            if (overfit_single_batch) dataloader_reset(&train_loader);
            dataloader_next_batch(&train_loader);
            sfnet_forward(&model, train_loader.inputs, B, T);
            sfnet_backward_and_reduce(&model, train_loader.inputs,
                                       train_loader.targets, grad_accum_steps, micro);
        }

        norm = sfnet_calculate_grad_norm(&model, &multi_gpu_config);
        update_detector(&loss_detector, (double)model.mean_loss);
        update_detector(&grad_norm_detector, (double)norm);

        float lr = get_learning_rate(&lr_sched, step);
        bool excessive_sanitize =
            g_last_sanitized_grad_count > (unsigned int)(model.num_parameters / 1000);
        if (excessive_sanitize) {
            printf0("warning: sanitized %u gradients at step %d, skipping update\n",
                    g_last_sanitized_grad_count, step + 1);
        } else if (!isfinite(norm)) {
            printf0("warning: non-finite grad norm at step %d, skipping update\n", step + 1);
        } else {
            float grad_scale = (grad_clip > 0.0f && norm > grad_clip)
                               ? grad_clip / norm : 1.0f;
            sfnet_update(&model, lr, 0.9f, 0.95f, 1e-8f, weight_decay,
                         grad_scale, step + 1, &multi_gpu_config);
        }

        cudaCheck(cudaEventRecord(ev_end));
        cudaCheck(cudaEventSynchronize(ev_end));
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, ev_start, ev_end));
        float tok_s = (float)(grad_accum_steps * ddp_world_size * B * T) /
                      (time_elapsed_ms * 1e-3f);
        float mfu = sfnet_estimate_mfu(&model, grad_accum_steps * B * T,
                                        time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %.6f | norm %.4f | lr %.2e | %.2f ms | %.0f tok/s | MFU %.2f%%\n",
                step + 1, num_iterations, model.mean_loss, norm, lr,
                time_elapsed_ms, tok_s, mfu * 100.0f);

        if (master_process && logfile) {
            FILE *lf = fopen(logfile, "a");
            if (lf) { fprintf(lf, "s:%d trl:%f\n", step, model.mean_loss); fclose(lf); }
        }

        if (master_process && output_dir && strlen(output_dir) > 0
            && step > 0 && step % 500 == 0) {
            char cp[512];
            snprintf(cp, sizeof(cp), "%s/sfnet_step%05d.bin", output_dir, step);
            sfnet_write_checkpoint(&model, cp);
        }
    }

    multi_gpu_config_free(&multi_gpu_config);
    dataloader_free(&train_loader);
    if (has_val) dataloader_free(&val_loader);
    cudaCheck(cudaFree(model.params_memory));
    cudaCheck(cudaFree(model.grads_memory));
    cudaCheck(cudaFree(model.acts_memory));
    cudaCheck(cudaFree(model.d_freqs_cis));
    cudaCheck(cudaFree(cublaslt_workspace));
    cudaCheck(cudaStreamDestroy(main_stream));
    if (tokenizer.init_ok) tokenizer_free(&tokenizer);
    printf0("Training complete.\n");
    return 0;
}
