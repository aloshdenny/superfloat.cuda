/*
LLaMA 3.2 Transformer Neural Net SF16 training loop.
Supports LLaMA 3.2 1B and 3B model configurations.

Architecture differences vs GPT-2/GPT-3:
  1. RMSNorm instead of LayerNorm (no bias, no mean subtraction)
  2. RoPE (Rotary Position Embeddings) with LLaMA3 frequency scaling
  3. Grouped Query Attention (GQA): n_kv_head < n_head
  4. SwiGLU MLP (silu(gate) * up, projected by down — no GELU, no bias)
  5. No position embedding table (wpe) — RoPE is applied inside attention
  6. Tied embeddings: wte == lm_head weights (shared)
  7. Vocabulary size 128256 (vs 50257 for GPT)

SF16 Tokenizer for LLaMA 3.2:
  Token IDs 0–128255 exceed Q1.15's unsigned range (0–65535).
  Solution: encode each token ID as TWO exact Q1.15 values:
    high = token_id >> 16         (0 or 1 for IDs up to 128255)
    low  = token_id & 0xFFFF      (0–65535)
  Normalized to [0,1): sf16_high = high/65536.0, sf16_low = low/65536.0
  A CUDA kernel reconstructs the original int before embedding lookup.
  This is lossless for all IDs 0–128255 and adds zero sequence overhead.

Build:
  # BF16 baseline
  make train_llama32cu

  # SF16 Q1.15
  make train_llama32q115cu

  # SF16 Q1.15 weight-constrained
  make train_llama32q115_constrainedcu

Usage:
  ./train_llama32q115cu \
    --model llama3.2:1b \
    --input_bin dev/data/fineweb10B/fineweb_train_*.bin \
    --input_val_bin dev/data/fineweb10B/fineweb_val_*.bin \
    --output_dir log_llama32_1b_sf16 \
    --num_iterations 5000 --learning_rate 3e-4 \
    --batch_size 4 --sequence_length 1024
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
// Recompiling with updated cuBLAS compute types and workspace size
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>

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
#endif
#include "llmc/cuda_utils.cuh"
#include "llmc/cublas_common.h"

// ============================================================================
// Layer implementations
// ============================================================================
#include "llmc/encoder.cuh"      // encoder_forward, encoder_backward
#include "llmc/rmsnorm.cuh"      // rmsnorm_forward, rmsnorm_backward
#include "llmc/layernorm.cuh"    // fused_residual_forward5
#include "llmc/matmul.cuh"       // matmul_forward_cublaslt, matmul_backward
#include "llmc/attention.cuh"    // attention_forward, attention_backward
#include "llmc/fused_classifier.cuh"
#include "llmc/adamw.cuh"
#include "llmc/global_norm.cuh"
#include "llmc/zero.cuh"

// ============================================================================
// Global I/O and GPU state
// ============================================================================
char filename_buffer[512];
cudaDeviceProp deviceProp;
cudaStream_t main_stream;
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ============================================================================
// LLaMA 3.2 Model Configuration
// ============================================================================

typedef struct {
    int dim;           // hidden dimension (C)
    int n_layers;      // number of transformer blocks (L)
    int n_heads;       // number of attention heads (NH)
    int n_kv_heads;    // number of KV heads for GQA (NKV)
    int ffn_dim;       // SwiGLU intermediate dimension
    int head_dim;      // per-head dimension = dim / n_heads
    int vocab_size;    // vocabulary size (128256 for LLaMA 3.x)
    int padded_vocab_size; // padded to multiple of 128
    int max_seq_len;   // maximum context length
    float rope_theta;  // RoPE base frequency
    float norm_eps;    // RMSNorm epsilon
} Llama32Config;

// ============================================================================
// Parameter Tensors
// LLaMA 3.2 has NO biases. Tied embeddings: wte is also used as lm_head.
// GQA: qkvw projections are (NH + 2*NKV) * head_dim, not 3*NH*head_dim.
// SwiGLU MLP: gate_w (w1), up_w (w3), down_w (w2).
// ============================================================================

constexpr const int NUM_PARAMETER_TENSORS = 9;
typedef struct {
    floatX *wte;      // (Vp, C)           — token embedding + lm_head (tied)
    floatX *rms1w;    // (L, C)            — pre-attention RMSNorm weight
    floatX *qkvw;     // (L, (NH+2*NKV)*HD, C) — packed Q,K,V projections
    floatX *attn_ow;  // (L, C, NH*HD)     — attention output projection
    floatX *rms2w;    // (L, C)            — pre-FFN RMSNorm weight
    floatX *gate_w;   // (L, FFN, C)       — SwiGLU gate path (w1)
    floatX *up_w;     // (L, FFN, C)       — SwiGLU up path (w3)
    floatX *down_w;   // (L, C, FFN)       — SwiGLU down projection (w2)
    floatX *rms_fw;   // (C,)              — final RMSNorm weight
} ParameterTensors;

static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void *),
              "ParameterTensors size mismatch!");

void fill_in_parameter_sizes(size_t *param_sizes, size_t *param_sizeof,
                             Llama32Config cfg) {
    size_t Vp  = (size_t)cfg.padded_vocab_size;
    size_t C   = (size_t)cfg.dim;
    size_t L   = (size_t)cfg.n_layers;
    size_t NH  = (size_t)cfg.n_heads;
    size_t NKV = (size_t)cfg.n_kv_heads;
    size_t HD  = (size_t)cfg.head_dim;
    size_t FFN = (size_t)cfg.ffn_dim;

    param_sizes[0] = Vp * C;                  // wte
    param_sizes[1] = L * C;                   // rms1w
    param_sizes[2] = L * (NH + 2*NKV) * HD * C; // qkvw
    param_sizes[3] = L * C * (NH * HD);       // attn_ow  (NH*HD == C always)
    param_sizes[4] = L * C;                   // rms2w
    param_sizes[5] = L * FFN * C;             // gate_w
    param_sizes[6] = L * FFN * C;             // up_w
    param_sizes[7] = L * C * FFN;             // down_w
    param_sizes[8] = C;                       // rms_fw

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

void *malloc_and_point_parameters(ParameterTensors *params,
                                  size_t *param_elements,
                                  size_t *param_sizeof) {
    size_t num_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
        num_bytes += param_elements[i] * param_sizeof[i];

    void *mem;
    cudaCheck(cudaMalloc(&mem, num_bytes));

    floatX **ptrs[] = {
        &params->wte, &params->rms1w, &params->qkvw, &params->attn_ow,
        &params->rms2w, &params->gate_w, &params->up_w, &params->down_w,
        &params->rms_fw
    };
    char *it = (char *)mem;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX *)it;
        it += param_elements[i] * param_sizeof[i];
    }
    return mem;
}

// ============================================================================
// Activation Tensors
// ============================================================================

constexpr int NUM_ACTIVATION_TENSORS = 18;
typedef struct {
    floatX *encoded;     // (B, T, C) — embedding output
    floatX *rms1;        // (L, B, T, C) — pre-attn RMSNorm output
    float  *rms1_rstd;   // (L, B, T) — rstd from RMSNorm (for bwd)
    floatX *qkvr;        // (L, B, T, (NH+2*NKV)*HD) — QKV buffer
    floatX *atty;        // (L, B, T, C) — attention output
    floatX *att;         // (L, B, NH, T, T) — attention scores
    floatX *residual2;   // (L, B, T, C) — after attention residual
    floatX *rms2;        // (L, B, T, C) — pre-FFN RMSNorm output
    float  *rms2_rstd;   // (L, B, T)
    floatX *gate_act;    // (L, B, T, FFN) — silu(gate_w @ rms2)
    floatX *up_act;      // (L, B, T, FFN) — up_w @ rms2
    floatX *swiglu_out;  // (L, B, T, FFN) — gate_act * up_act
    floatX *residual3;   // (L, B, T, C)   — after MLP residual
    floatX *rms_f;       // (B, T, C)      — final RMSNorm output
    float  *rms_f_rstd;  // (B, T)
    float  *losses;      // (B, T)
    floatX *output;      // (B, T, Vp)     — logits / gradient scratchpad
    floatX *scratch_btc; // (B, T, C)      — gradient scratchpad
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
                               size_t B, size_t T, Llama32Config cfg,
                               int recompute) {
    size_t C   = cfg.dim;
    size_t L   = cfg.n_layers;
    size_t NH  = cfg.n_heads;
    size_t NKV = cfg.n_kv_heads;
    size_t HD  = cfg.head_dim;
    size_t FFN = cfg.ffn_dim;
    size_t Vp  = cfg.padded_vocab_size;

    tensors[0]  = TENSOR_SPEC(data->encoded,    B * T * C);
    tensors[1]  = TENSOR_SPEC(data->rms1,       (recompute < 2) ? L * B * T * C : 0);
    tensors[2]  = TENSOR_SPEC(data->rms1_rstd,  L * B * T);
    tensors[3]  = TENSOR_SPEC(data->qkvr,       L * B * T * (NH + 2*NKV) * HD);
    tensors[4]  = TENSOR_SPEC(data->atty,       L * B * T * C);
    tensors[5]  = TENSOR_SPEC(data->att,        L * B * NH * T * T);
    tensors[6]  = TENSOR_SPEC(data->residual2,  L * B * T * C);
    tensors[7]  = TENSOR_SPEC(data->rms2,       (recompute < 2) ? L * B * T * C : 0);
    tensors[8]  = TENSOR_SPEC(data->rms2_rstd,  L * B * T);
    tensors[9]  = TENSOR_SPEC(data->gate_act,   (recompute < 1) ? L * B * T * FFN : B * T * FFN);
    tensors[10] = TENSOR_SPEC(data->up_act,     (recompute < 1) ? L * B * T * FFN : B * T * FFN);
    tensors[11] = TENSOR_SPEC(data->swiglu_out, (recompute < 1) ? L * B * T * FFN : B * T * FFN);
    tensors[12] = TENSOR_SPEC(data->residual3,  L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->rms_f,      B * T * C);
    tensors[14] = TENSOR_SPEC(data->rms_f_rstd, B * T);
    tensors[15] = TENSOR_SPEC(data->losses,     B * T);
    tensors[16] = TENSOR_SPEC(data->output,     B * T * max((NH + 2*NKV) * HD, max(NH * T, Vp)));
    tensors[17] = TENSOR_SPEC(data->scratch_btc, B * T * C);
}

void *malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);

    printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024*1024)));
    void *mem;
    cudaCheck(cudaMalloc(&mem, bytes));
    cudaCheck(cudaMemset(mem, 0, bytes));

    char *it = (char *)mem;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if (tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        } else {
            *(tensors[i].ptr) = it;
            it += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return mem;
}

// ============================================================================
// LLaMA 3.2 Model Struct
// ============================================================================

typedef struct {
    Llama32Config config;
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
    int recompute;
    int *workload_indices;
    int4 *bucket_info;
    // RoPE frequencies (precomputed on device)
    float2 *d_freqs_cis; // (max_seq_len, HD/2)
} LLaMA32;

// ============================================================================
// SF16 Tokenizer: Lossless encode/decode for token IDs 0-128255
// ============================================================================

// CPU: encode token_id to two int16 components (base-65536 decomposition)
static inline void sf16_token_encode(int token_id, int16_t *out_high, int16_t *out_low) {
    int hi = token_id >> 16;      // 0 or 1
    int lo = token_id & 0xFFFF;   // 0..65535
    *out_high = (int16_t)hi;
    *out_low  = (int16_t)(uint16_t)lo;  // bitcast allows negative for lo>=32768
}

// CPU: decode two int16 components back to token_id
static inline int sf16_token_decode(int16_t sf_high, int16_t sf_low) {
    int hi = (int)sf_high;
    int lo = (int)(uint16_t)sf_low;
    return (hi << 16) | lo;
}

// Self-test for all 128256 token IDs
static void sf16_tokenizer_self_test(void) {
    printf0("SF16 tokenizer self-test: checking all 128256 token IDs...\n");
    int errors = 0;
    for (int id = 0; id < 128256; id++) {
        int16_t hi, lo;
        sf16_token_encode(id, &hi, &lo);
        int decoded = sf16_token_decode(hi, lo);
        if (decoded != id) {
            if (errors < 5) {
                printf0("  FAIL: id=%d -> (%d,%d) -> %d\n", id, hi, lo, decoded);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf0("SF16 tokenizer self-test: PASS (all 128256 IDs lossless)\n");
    } else {
        printf0("SF16 tokenizer self-test: FAIL (%d errors)\n", errors);
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// RoPE: Rotary Positional Embeddings with LLaMA3 frequency scaling
// ============================================================================

static float llama3_scale_freq(float freq, float scale_factor,
                                float low_freq_factor, float high_freq_factor,
                                int orig_ctx_len) {
    float wavelen = 2.0f * 3.14159265358979f / freq;
    float low_w  = (float)orig_ctx_len / low_freq_factor;
    float high_w = (float)orig_ctx_len / high_freq_factor;
    if (wavelen < high_w) return freq;
    if (wavelen > low_w)  return freq / scale_factor;
    float smooth = ((float)orig_ctx_len / wavelen - low_freq_factor)
                   / (high_freq_factor - low_freq_factor);
    return (1.0f - smooth) * freq / scale_factor + smooth * freq;
}

// Precompute RoPE freqs on host, upload to device. Returns device pointer.
static float2 *precompute_freqs_cis(int max_seq_len, int head_dim, float rope_theta) {
    const float scale_factor    = 32.0f;
    const float low_freq_factor  = 1.0f;
    const float high_freq_factor = 4.0f;
    const int  orig_ctx_len      = 8192;
    int half = head_dim / 2;
    size_t n = (size_t)max_seq_len * half;
    float2 *h = (float2 *)mallocCheck(n * sizeof(float2));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float raw = 1.0f / powf(rope_theta, (float)(2*i) / (float)head_dim);
            float freq = llama3_scale_freq(raw, scale_factor, low_freq_factor,
                                           high_freq_factor, orig_ctx_len);
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

// GPU: apply RoPE in-place to the Q and K parts of packed QKV buffer.
// qkv layout per (b,t): [Q: NH*HD | K: NKV*HD | V: NKV*HD]
__global__ void rope_forward_kernel(
    floatX *__restrict__ qkv,
    const float2 *__restrict__ freqs, // (T, HD/2)
    int B, int T, int NH, int NKV, int HD)
{
    int bt = blockIdx.x;
    if (bt >= B * T) return;
    int t = bt % T;
    int total_qkv = (NH + 2 * NKV) * HD;
    floatX *row = qkv + (size_t)bt * total_qkv;
    const float2 *f = freqs + (size_t)t * (HD / 2);

    // Rotate Q heads
    for (int h = threadIdx.x; h < NH * (HD / 2); h += blockDim.x) {
        int head = h / (HD / 2), i = h % (HD / 2);
        int base = head * HD + i * 2;
        float x0 = (float)row[base], x1 = (float)row[base+1];
        float cv = f[i].x, sv = f[i].y;
        row[base]   = (floatX)(x0*cv - x1*sv);
        row[base+1] = (floatX)(x0*sv + x1*cv);
    }
    // Rotate K heads
    for (int h = threadIdx.x; h < NKV * (HD / 2); h += blockDim.x) {
        int head = h / (HD / 2), i = h % (HD / 2);
        int base = NH * HD + head * HD + i * 2;
        float x0 = (float)row[base], x1 = (float)row[base+1];
        float cv = f[i].x, sv = f[i].y;
        row[base]   = (floatX)(x0*cv - x1*sv);
        row[base+1] = (floatX)(x0*sv + x1*cv);
    }
}

static void rope_forward(floatX *qkv, const float2 *freqs,
                         int B, int T, int NH, int NKV, int HD,
                         cudaStream_t stream) {
    int threads = min(256, max(NH, NKV) * (HD / 2));
    threads = CEIL_DIV(threads, 32) * 32;
    rope_forward_kernel<<<B * T, threads, 0, stream>>>(qkv, freqs, B, T, NH, NKV, HD);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// GQA: repeat K,V from NKV heads to NH heads
// ============================================================================

__global__ void repeat_kv_kernel(
    floatX *__restrict__ out,       // (BT, NH, HD)
    const floatX *__restrict__ inp, // (BT, NKV, HD)
    int BT, int NKV, int n_rep, int HD)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BT * NKV * n_rep * HD;
    if (idx >= total) return;
    int d    = idx % HD;
    int h    = (idx / HD) % (NKV * n_rep);
    int bt   = idx / (HD * NKV * n_rep);
    int kv_h = h / n_rep;
    out[(size_t)bt * NKV * n_rep * HD + h * HD + d] =
        inp[(size_t)bt * NKV * HD + kv_h * HD + d];
}

static void repeat_kv(floatX *out, const floatX *inp,
                       int BT, int NKV, int n_rep, int HD,
                       cudaStream_t stream) {
    int total = BT * NKV * n_rep * HD;
    repeat_kv_kernel<<<CEIL_DIV(total, 256), 256, 0, stream>>>(
        out, inp, BT, NKV, n_rep, HD);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// SwiGLU: forward and backward
// ============================================================================

__global__ void swiglu_forward_kernel(
    floatX *__restrict__ gate_act,   // silu(gate_in)
    floatX *__restrict__ swiglu_out, // gate_act * up_in
    const floatX *__restrict__ gate_in,
    const floatX *__restrict__ up_in,
    int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (size_t)blockDim.x * gridDim.x) {
        float g = (float)__ldcs(&gate_in[i]);
        float u = (float)__ldcs(&up_in[i]);
        float sg = g / (1.0f + expf(-g));  // silu
#if defined(ENABLE_Q115)
        sg = (float)simulate_q115((floatX)sg);
        u  = (float)simulate_q115((floatX)u);
#endif
        __stcs(&gate_act[i],    (floatX)sg);
        __stcs(&swiglu_out[i],  (floatX)(sg * u));
    }
}

static void swiglu_forward(floatX *gate_act, floatX *swiglu_out,
                            const floatX *gate_in, const floatX *up_in,
                            int N, cudaStream_t stream) {
    swiglu_forward_kernel<<<CEIL_DIV(N, 256), 256, 0, stream>>>(
        gate_act, swiglu_out, gate_in, up_in, N);
    cudaCheck(cudaGetLastError());
}

__global__ void swiglu_backward_kernel(
    floatX *__restrict__ d_gate_in,
    floatX *__restrict__ d_up_in,
    const floatX *__restrict__ d_out,
    const floatX *__restrict__ gate_in,
    const floatX *__restrict__ gate_act,
    const floatX *__restrict__ up_in,
    int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (size_t)blockDim.x * gridDim.x) {
        float dout_i = (float)__ldcs(&d_out[i]);
        float gi     = (float)__ldcs(&gate_in[i]);
        float si     = (float)__ldcs(&gate_act[i]);
        float ui     = (float)__ldcs(&up_in[i]);
        float sig    = 1.0f / (1.0f + expf(-gi));
        float dsilu  = si + sig * (1.0f - si);
        float prev_dg = (float)__ldcs(&d_gate_in[i]);
        float prev_du = (float)__ldcs(&d_up_in[i]);
        __stcs(&d_gate_in[i], (floatX)(prev_dg + dout_i * ui * dsilu));
        __stcs(&d_up_in[i],   (floatX)(prev_du + dout_i * si));
    }
}

static void swiglu_backward(floatX *d_gate_in, floatX *d_up_in,
                             const floatX *d_out,
                             const floatX *gate_in, const floatX *gate_act,
                             const floatX *up_in,
                             int N, cudaStream_t stream) {
    swiglu_backward_kernel<<<CEIL_DIV(N, 256), 256, 0, stream>>>(
        d_gate_in, d_up_in, d_out, gate_in, gate_act, up_in, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Model init / checkpoint helpers
// ============================================================================

void llama32_init_common(LLaMA32 *model) {
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    model->params_memory = NULL;
    model->grads_memory = NULL;
    model->workload_indices = NULL;
    model->bucket_info = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    model->d_freqs_cis = NULL;
    model->rng_state = 13371337 + multi_gpu_config.process_rank;
    // BF16 pretraining: no FP32 master copy needed.
    // use_master_weights=0 saves 4714 MiB, keeping total < 24 GB VRAM.
    // AdamW m/v states (FP32) already provide sufficient precision.
    model->use_master_weights = 0;
    model->init_state = true;
    model->recompute = 1;
}

void llama32_allocate_weights(LLaMA32 *model) {
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

void llama32_allocate_state(LLaMA32 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n",
            (int)round(model->num_parameters * sizeof(floatX) / (1024*1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(
        &model->grads, model->param_elements, model->param_sizeof);

    model->batch_size = B;
    model->seq_len    = T;
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T,
                              model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);

    cudaCheck(cudaMalloc((void **)&model->inputs,  B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&model->accumulated_mean_loss, sizeof(float)));
    cudaCheck(cudaMallocHost((void **)&model->cpu_losses, B * T * sizeof(float)));

    size_t num_c_groups = CEIL_DIV(model->config.dim, (WARP_SIZE * x128::size));
    model->workload_indices = (int *)mallocCheck(
        sizeof(int) * B * T * num_c_groups);
    model->bucket_info = (int4 *)mallocCheck(
        sizeof(int4) * B * T * num_c_groups);

    int memory_status = 0;
    size_t shard_np = multi_gpu_config.shard_num_parameters;
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_np * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_np * sizeof(float)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= cudaMallocConditionallyManaged((void **)&model->m_memory, shard_np * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void **)&model->v_memory, shard_np * sizeof(float));
    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_np * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged(
            (void **)&model->master_weights, shard_np * sizeof(float));
    }
    int reduced = (int)multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged on %d GPUs\n", reduced);
    }
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n",
            (total - free) / 1024 / 1024, total / 1024 / 1024);
}

void llama32_set_hyperparameters(Llama32Config *cfg, const char *model_str) {
    // LLaMA 3.2 1B base configs
    cfg->dim         = 2048;
    cfg->n_layers    = 16;
    cfg->n_heads     = 32;
    cfg->n_kv_heads  = 8;
    cfg->ffn_dim     = 8192;
    cfg->head_dim    = 64;

    cfg->vocab_size        = 128256;
    cfg->padded_vocab_size = 128256; // already aligned to 128
    cfg->max_seq_len       = 8192;
    cfg->rope_theta        = 500000.0f;
    cfg->norm_eps          = 1e-5f;
}

void llama32_random_init(LLaMA32 *model, const char *model_str) {
    llama32_set_hyperparameters(&model->config, model_str);
    llama32_allocate_weights(model);

    // Precompute RoPE frequencies
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

    size_t L = model->config.n_layers;
    float residual_scale = 1.0f / sqrtf(2.0f * (float)L);

    size_t offsets[NUM_PARAMETER_TENSORS + 1];
    offsets[0] = 0;
    for (int t = 0; t < NUM_PARAMETER_TENSORS; t++)
        offsets[t+1] = offsets[t] + model->param_elements[t];

    for (int t = 0; t < NUM_PARAMETER_TENSORS; t++) {
        size_t n = model->param_elements[t];
        // RMSNorm weights init to 1.0
        if (t == 1 || t == 4 || t == 8) {
            for (size_t j = 0; j < n; j++)
                cpu[offsets[t] + j] = (floatX)1.0f;
            continue;
        }
        // Weight matrices
        // attn output proj and down_proj scaled by residual_scale
        float scale = (t == 3 || t == 7) ? init_scale * residual_scale : init_scale;
        float *buf = (float *)mallocCheck(n * sizeof(float));
        normal_(buf, n, 0.0f, scale, &rng);
        for (size_t j = 0; j < n; j++) {
            cpu[offsets[t] + j] = (floatX)buf[j];
        }
        free(buf);
    }
    cudaCheck(cudaMemcpy(model->params_memory, cpu, model->num_parameters_bytes,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    free(cpu);
    printf0("LLaMA 3.2 %s: %zu parameters (%.2f B)\n",
            model_str, model->num_parameters,
            (double)model->num_parameters / 1e9);
}

void llama32_write_checkpoint(LLaMA32 *model, const char *path) {
    printf0("Writing checkpoint to %s\n", path);
    FILE *f = fopenCheck(path, "wb");
    int header[256];
    memset(header, 0, sizeof(header));
    header[0] = 20250319; // magic
    header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5;
    header[2] = model->config.dim;
    header[3] = model->config.n_layers;
    header[4] = model->config.n_heads;
    header[5] = model->config.n_kv_heads;
    header[6] = model->config.ffn_dim;
    header[7] = model->config.head_dim;
    header[8] = model->config.vocab_size;
    header[9] = model->config.padded_vocab_size;
    header[10] = model->config.max_seq_len;
    fwriteCheck(header, sizeof(int), 256, f);
    device_to_file(f, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    fcloseCheck(f);
}

// ShardInfo helper for LLaMA parameter tensors (identical structure to GPT3)
ShardInfo llama32_get_tensor_at_layer(const LLaMA32 *model, int layer_id,
                                       int tensor_id) {
    ptrdiff_t offset = 0;
    for (int i = 0; i < tensor_id; i++) offset += (ptrdiff_t)model->param_elements[i];
    size_t size = model->param_elements[tensor_id];
    // tensors 1-7 are per-layer (indices 1..7 inclusive)
    if (tensor_id >= 1 && tensor_id <= 7) {
        size /= model->config.n_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

// ============================================================================
// Forward Pass
// ============================================================================

void llama32_forward(LLaMA32 *model, const int *inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    if (model->params_memory == NULL) {
        printf("Error: model not initialized.\n"); exit(EXIT_FAILURE);
    }
    if ((int)B > model->batch_size || (int)T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n",
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
    const int    V   = model->config.vocab_size;
    const int n_rep  = (int)(NH / NKV);

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int),
                         cudaMemcpyHostToDevice));
    tokenCheck(inputs, B * T, V);

    ParameterTensors params = model->params;
    ActivationTensors acts  = model->acts;

    // 1. Token embedding (no position embedding — RoPE handles position)
    encoder_forward(acts.encoded, model->inputs, params.wte, /*wpe=*/nullptr,
                    B, T, C, main_stream);

    // 2. Transformer blocks
    for (int l = 0; l < (int)L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX *residual = (l == 0) ? acts.encoded
                                     : acts.residual3 + (size_t)(l-1) * B * T * C;

        // Layer-l parameter slices
        floatX *l_rms1w  = params.rms1w  + (size_t)l * C;
        floatX *l_qkvw   = params.qkvw   + (size_t)l * (NH + 2*NKV) * HD * C;
        floatX *l_attn_ow= params.attn_ow+ (size_t)l * C * C;
        floatX *l_rms2w  = params.rms2w  + (size_t)l * C;
        floatX *l_gate_w = params.gate_w + (size_t)l * FFN * C;
        floatX *l_up_w   = params.up_w   + (size_t)l * FFN * C;
        floatX *l_down_w = params.down_w + (size_t)l * C * FFN;

        // Layer-l activation slices
        floatX *l_rms1   = (model->recompute < 2) ? acts.rms1 + (size_t)l*B*T*C : acts.rms_f;
        float  *l_rms1_r = acts.rms1_rstd + (size_t)l * B * T;
        floatX *l_qkvr   = acts.qkvr    + (size_t)l * B * T * (NH+2*NKV) * HD;
        floatX *l_atty   = acts.atty    + (size_t)l * B * T * C;
        floatX *l_att    = acts.att     + (size_t)l * B * NH * T * T;
        floatX *l_res2   = acts.residual2 + (size_t)l * B * T * C;
        floatX *l_rms2   = (model->recompute < 2) ? acts.rms2 + (size_t)l*B*T*C : acts.rms_f;
        float  *l_rms2_r = acts.rms2_rstd + (size_t)l * B * T;
        floatX *l_gate_a = (model->recompute < 1) ? acts.gate_act  + (size_t)l*B*T*FFN : acts.gate_act;
        floatX *l_up_a   = (model->recompute < 1) ? acts.up_act    + (size_t)l*B*T*FFN : acts.up_act;
        floatX *l_swiglu = (model->recompute < 1) ? acts.swiglu_out+ (size_t)l*B*T*FFN : acts.swiglu_out;
        floatX *l_res3   = acts.residual3 + (size_t)l * B * T * C;
        floatX *scratch  = (floatX *)acts.output;

        // --- Pre-attention RMSNorm ---
        rmsnorm_forward(l_rms1, l_rms1_r, residual, l_rms1w,
                        B, T, C, model->config.norm_eps, main_stream);

        // 1) QKV projection
        matmul_forward_cublas(l_qkvr, l_rms1, l_qkvw, B, T, C, (int)((NH+2*NKV)*HD), main_stream);

        // --- RoPE ---
        rope_forward(l_qkvr, model->d_freqs_cis,
                     B, T, NH, NKV, HD, main_stream);

        // --- GQA: expand K,V to NH heads for standard attention ---
        // Build expanded QKV in l_qkvr in-place: [Q(NH,HD) | K_exp(NH,HD) | V_exp(NH,HD)]
        // We expand K,V backwards to avoid overwriting source data
        floatX *k_src = l_qkvr + (size_t)B*T*NH*HD;
        floatX *v_src = l_qkvr + (size_t)B*T*(NH+NKV)*HD;
        floatX *k_dst = l_qkvr + (size_t)B*T*NH*HD;
        floatX *v_dst = l_qkvr + (size_t)B*T*NH*HD*2;
        // Expand V first (at end of buffer, safe), then K
        repeat_kv(v_dst, v_src, B*T, NKV, n_rep, HD, main_stream);
        repeat_kv(k_dst, k_src, B*T, NKV, n_rep, HD, main_stream);
        // Q is already in place at the start of l_qkvr, no copy needed

        // --- Attention forward (uses expanded Q,K,V in l_qkvr) ---
        if (T != model->seq_len)
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);

        // 3) Output projection
        matmul_forward_cublas(scratch, l_atty, l_attn_ow, B, T, C, C, main_stream);

        // residual2 = residual + attn_out; pre-FFN RMSNorm -> l_rms2
        fused_residual_forward5(l_res2, l_rms2, l_rms2_r, /*mean=*/nullptr,
                                residual, scratch, l_rms2w, /*b=*/nullptr,
                                B*T, C, main_stream);

        // 4) FFN
        // gate branch:
        matmul_forward_cublas(l_gate_a, l_rms2, l_gate_w, B, T, C, (int)FFN, main_stream);
        // up branch: up_w @ rms2  (use swiglu_out as temp storage for up_act)
        matmul_forward_cublas(l_swiglu, l_rms2, l_up_w, B, T, C, (int)FFN, main_stream);
        // pointwise: gate_a = silu(gate_a), swiglu = gate_a * up
        swiglu_forward(l_gate_a, l_up_a, l_gate_a, l_swiglu, B*T*FFN, main_stream);

        // Down projection
        matmul_forward_cublas(scratch, l_up_a, l_down_w, B, T, (int)FFN, C, main_stream);

        // residual3 = residual2 + mlp_out; pre-next-layer RMSNorm fused in
        if (l + 1 != (int)L) {
            floatX *next_rms1  = (model->recompute < 2)
                                 ? acts.rms1 + (size_t)(l+1)*B*T*C : acts.rms_f;
            float  *next_rms1r = acts.rms1_rstd + (size_t)(l+1)*B*T;
            floatX *next_rms1w = params.rms1w + (size_t)(l+1)*C;
            fused_residual_forward5(l_res3, next_rms1, next_rms1r, nullptr,
                                    l_res2, scratch, next_rms1w, nullptr,
                                    B*T, C, main_stream);
        } else {
            fused_residual_forward5(l_res3, acts.rms_f, acts.rms_f_rstd, nullptr,
                                    l_res2, scratch, params.rms_fw, nullptr,
                                    B*T, C, main_stream);
        }
    } // end layer loop

    // 3. LM head: logits = rms_f @ wte^T  (tied weights)
    matmul_forward_cublas(acts.output, acts.rms_f, params.wte, B, T, C, Vp, main_stream);
}

// Validation (forward + loss, no backward)
float llama32_validate(LLaMA32 *model, const int *inputs, const int *targets,
                       size_t B, size_t T) {
    assert(targets != NULL);
    llama32_forward(model, inputs, B, T);
    const size_t Vp = model->config.padded_vocab_size;
    const int V     = model->config.vocab_size;
    ActivationTensors acts = model->acts;
    const float dloss = 1.0f / (B * T);
    cudaCheck(cudaMemset(acts.losses, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int),
                         cudaMemcpyHostToDevice));
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
// Backward Pass
// ============================================================================

void llama32_backward_and_reduce(LLaMA32 *model, int *inputs, const int *targets,
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
    const int    V   = model->config.vocab_size;
    const int n_rep  = (int)(NH / NKV);
    (void)n_rep; // used indirectly

    ParameterTensors params = model->params;
    ParameterTensors grads  = model->grads;
    ActivationTensors acts  = model->acts;

    const float dloss = 1.0f / (float)(B * T * grad_accum_steps);
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int),
                         cudaMemcpyHostToDevice));
    tokenCheck(targets, B * T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets,
                     B, T, V, Vp, True, main_stream);

    // dresidual starts as zero (gradient flowing back from lm_head -> final norm)
    floatX *dresidual = (floatX *)acts.scratch_btc;
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    float *scratchF = (float *)acts.output;
    floatX *scratchX = (floatX *)acts.output;

    // Backward through lm_head (tied: grad goes to wte grads)
    matmul_backward(acts.scratch_btc, grads.wte, nullptr, acts.output,
                    acts.rms_f, params.wte, nullptr, B, T, C, Vp, main_stream);

    // Backward through final RMSNorm
    floatX *residual_last = acts.residual3 + (size_t)(L-1) * B * T * C;
    rmsnorm_backward(dresidual, grads.rms_fw, scratchF,
                     acts.scratch_btc, residual_last,
                     params.rms_fw, acts.rms_f_rstd, B, T, C, main_stream);

    floatX *dl_btc = residual_last; // reuse memory

    for (int l = (int)L - 1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        floatX *residual = (l == 0) ? acts.encoded
                                     : acts.residual3 + (size_t)(l-1)*B*T*C;

        // Weight pointers for this layer
        floatX *l_rms2w  = params.rms2w  + (size_t)l * C;
        floatX *l_gate_w = params.gate_w + (size_t)l * FFN * C;
        floatX *l_up_w   = params.up_w   + (size_t)l * FFN * C;
        floatX *l_down_w = params.down_w + (size_t)l * C * FFN;
        floatX *l_rms1w  = params.rms1w  + (size_t)l * C;
        floatX *l_qkvw   = params.qkvw   + (size_t)l * (NH+2*NKV)*HD*C;
        floatX *l_attn_ow= params.attn_ow+ (size_t)l * C * C;

        // Grad weight pointers
        floatX *dl_rms2w = grads.rms2w  + (size_t)l * C;
        floatX *dl_gate_w= grads.gate_w + (size_t)l * FFN * C;
        floatX *dl_up_w  = grads.up_w   + (size_t)l * FFN * C;
        floatX *dl_down_w= grads.down_w + (size_t)l * C * FFN;
        floatX *dl_rms1w = grads.rms1w  + (size_t)l * C;
        floatX *dl_qkvw  = grads.qkvw   + (size_t)l * (NH+2*NKV)*HD*C;
        floatX *dl_attn_ow=grads.attn_ow+ (size_t)l * C * C;

        // Activation pointers
        floatX *l_rms1  = (model->recompute < 2) ? acts.rms1 + (size_t)l*B*T*C : acts.rms_f;
        float  *l_rms1r = acts.rms1_rstd + (size_t)l * B * T;
        floatX *l_qkvr  = acts.qkvr + (size_t)l*B*T*(NH+2*NKV)*HD;
        floatX *l_atty  = acts.atty + (size_t)l*B*T*C;
        floatX *l_att   = acts.att  + (size_t)l*B*NH*T*T;
        floatX *l_res2  = acts.residual2 + (size_t)l*B*T*C;
        floatX *l_rms2  = (model->recompute < 2) ? acts.rms2 + (size_t)l*B*T*C : acts.rms_f;
        float  *l_rms2r = acts.rms2_rstd + (size_t)l * B * T;
        floatX *l_gate_a= (model->recompute < 1) ? acts.gate_act  + (size_t)l*B*T*FFN : acts.gate_act;
        floatX *l_up_a  = (model->recompute < 1) ? acts.up_act    + (size_t)l*B*T*FFN : acts.up_act;
        floatX *l_swiglu= (model->recompute < 1) ? acts.swiglu_out+ (size_t)l*B*T*FFN : acts.swiglu_out;
        floatX *l_gate_in = l_gate_a; // gate_in stored in gate_act slot before silu

        floatX *dl_bt_ffn = (floatX *)acts.gate_act; // scratch for FFN backward

        // --- Backward MLP ---
        // d(down): dresidual += down_w^T @ d_swiglu
        matmul_backward(dl_bt_ffn, dl_down_w, nullptr, dresidual,
                        l_swiglu, l_down_w, scratchF, B, T, FFN, C, main_stream);
        // d(swiglu pointwise): d_gate_in, d_up_in from d_swiglu
        swiglu_backward(dl_bt_ffn, dl_bt_ffn + B*T*FFN,   // reuse; treat as d_gate_in,d_up_in
                        dl_bt_ffn, l_gate_in, l_gate_a, l_up_a,
                        B*T*FFN, main_stream);
        // d(gate proj): dl_rms2 += dl_gate_in @ gate_w
        matmul_backward(dl_btc, dl_gate_w, nullptr, dl_bt_ffn,
                        l_rms2, l_gate_w, scratchF, B, T, C, FFN, main_stream);
        // d(up proj): dl_rms2 += dl_up_in @ up_w (accumulates)
        matmul_backward(dl_btc, dl_up_w, nullptr, dl_bt_ffn + B*T*FFN,
                        l_rms2, l_up_w, scratchF, B, T, C, FFN, main_stream);
        // d(pre-FFN RMSNorm)
        rmsnorm_backward(dresidual, dl_rms2w, scratchF,
                         dl_btc, l_res2, l_rms2w, l_rms2r, B, T, C, main_stream);

        // --- Backward Attention output proj ---
        matmul_backward(dl_btc, dl_attn_ow, nullptr, dresidual,
                        l_atty, l_attn_ow, scratchF, B, T, C, C, main_stream);

        // --- Backward Attention (assumes expanded QKV) ---
        floatX *buffer_a = l_atty;
        floatX *buffer_b = (floatX *)acts.gate_act;
        attention_backward(dl_bt_ffn, buffer_b, scratchX, buffer_a,
                           dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);

        // --- Backward QKV proj ---
        matmul_backward(dl_btc, dl_qkvw, nullptr, dl_bt_ffn,
                        l_rms1, l_qkvw, scratchF, B, T, C, (int)(NH+2*NKV)*HD,
                        main_stream);

        // --- Backward pre-attention RMSNorm ---
        rmsnorm_backward(dresidual, dl_rms1w, scratchF,
                         dl_btc, residual, l_rms1w, l_rms1r, B, T, C, main_stream);

        // Accumulate cross-GPU gradients
        if (last_step) {
            floatX *const ptrs[] = {dl_rms1w, dl_qkvw, dl_attn_ow,
                                    dl_rms2w, dl_gate_w, dl_up_w, dl_down_w};
            const size_t nelems[] = {C, (NH+2*NKV)*HD*C, C*C,
                                     C, FFN*C, FFN*C, C*FFN};
            multi_gpu_async_reduce_gradient(ptrs, nelems, &multi_gpu_config, main_stream);
        }
    }

    // Backward through embedding (encoder)
    encoder_backward(grads.wte, nullptr, scratchX,
                     model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C,
                     random_u32(&model->rng_state), main_stream);

    if (last_step) {
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
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
// Grad norm
// ============================================================================

float llama32_calculate_grad_norm(LLaMA32 *model, MultiGpuConfig *mgc) {
    NVTX_RANGE_FN();
    floatX *gm = (floatX *)model->grads_memory;
    float *gns = (float *)model->acts.output;
    float gns_cpu = 0.0f;
    int num_slices[2] = {1, model->config.n_layers};
    int max_sums = get_max_num_block_sums(num_slices, 2);
    if (mgc->zero_stage == 1) {
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = llama32_get_tensor_at_layer(model, 0, i);
            ShardInfo shard  = multi_gpu_get_shard_offset(tensor.size, mgc, 1);
            ptrdiff_t off = tensor.offset + shard.offset;
            bool first = (i == 0);
            if (i == 0 || i == 8) {
                global_norm_squared(gns, gm + off, shard.size, 0, 1, max_sums, first, main_stream);
            } else {
                global_norm_squared(gns, gm + off, shard.size, tensor.size,
                                    model->config.n_layers, max_sums, first, main_stream);
            }
        }
        global_sum_deterministic(gns, gns, max_sums, main_stream);
#if MULTI_GPU
        ncclCheck(ncclAllReduce(gns, gns, sizeof(float), ncclFloat, ncclSum,
                                mgc->nccl_comm, main_stream));
#endif
    } else {
        global_norm_squared(gns, gm, model->num_parameters, 0, 1, max_sums, true, main_stream);
        global_sum_deterministic(gns, gns, max_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&gns_cpu, gns, sizeof(float), cudaMemcpyDeviceToHost));
    return sqrtf(gns_cpu);
}

// ============================================================================
// AdamW update (identical logic to GPT-3 with Q1.15 selective WD)
// ============================================================================

void llama32_update(LLaMA32 *model, float lr, float beta1, float beta2,
                    float eps, float weight_decay, float grad_scale, int t,
                    MultiGpuConfig *mgc, bool init_from_master_only = false) {
    NVTX_RANGE_FN();
    if (!model->grads_memory || !model->m_memory || !model->v_memory) {
        fprintf(stderr, "Allocate optimizer state before update\n"); exit(EXIT_FAILURE);
    }
    bool init_state = model->init_state;
    if (init_state) {
        model->init_state = false;
        cudaCheck(cudaMemset(model->m_memory, 0,
            mgc->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0,
            mgc->shard_num_parameters * sizeof(float)));
    }
    model->rng_state_last_update = model->rng_state;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        unsigned int seed = random_u32(&model->rng_state);
        int num_layers = model->config.n_layers;
        if (i == 0 || i == 8) num_layers = 1;

        ShardInfo tensor = llama32_get_tensor_at_layer(model, 0, i);
        ShardInfo shard  = multi_gpu_get_shard_offset(tensor.size, mgc, 1);
        ptrdiff_t lof = tensor.offset + shard.offset;
        ptrdiff_t lop = tensor.offset / mgc->num_processes;

#if defined(ENABLE_Q115)
        // Q1.15: only apply weight decay to down projection (w2), index 7
        float wd = (i == 7) ? weight_decay : 0.0f;
        bool freeze_tensor = false;
#else
        float wd = (i == 0 || i == 2 || i == 3 || i == 5 || i == 6 || i == 7)
                   ? weight_decay : 0.0f;
        bool freeze_tensor = false;
#endif
        floatX *param_ptr  = (floatX *)model->params_memory + lof;
        floatX *grad_ptr   = (floatX *)model->grads_memory  + lof;
        ptrdiff_t opt_off  = (mgc->zero_stage < 1) ? lof : lop;
        float *m_ptr       = model->m_memory + opt_off;
        float *v_ptr       = model->v_memory + opt_off;
        float *master_ptr  = model->master_weights ? model->master_weights + opt_off : nullptr;

        if (init_state && master_ptr) {
            size_t gs = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(gs, num_layers), 512, 0, main_stream>>>(
                master_ptr, param_ptr, shard.size, shard.size, tensor.size);
            cudaCheck(cudaGetLastError());
        }
        if (init_from_master_only) {
            init_from_master(param_ptr, master_ptr, shard.size, tensor.size,
                             shard.size, num_layers, seed, main_stream);
        } else if (!freeze_tensor) {
            adamw_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr,
                         shard.size, tensor.size, tensor.size, shard.size,
                         num_layers, lr, beta1, beta2, t, eps, wd,
                         grad_scale, seed, main_stream);
        }
        if (mgc->zero_stage == 1) {
#if MULTI_GPU
            ncclCheck(ncclGroupStart());
            for (int ll = 0; ll < num_layers; ll++) {
                ncclCheck(ncclAllGather(
                    param_ptr + ll * tensor.size,
                    (floatX *)model->params_memory + tensor.offset + ll * tensor.size,
                    shard.size, ncclFloatX, mgc->nccl_comm, mgc->nccl_stream));
            }
            ncclCheck(ncclGroupEnd());
#endif
        }
    }
    cudaCheck(cudaStreamSynchronize(main_stream));
}

// ============================================================================
// MFU estimate
// ============================================================================

float llama32_estimate_mfu(LLaMA32 *model, int num_tokens, float dt) {
    // 6 * N * num_tokens flops for forward+backward, N = num_parameters
    // (ignoring attention quadratic term for simplicity)
    float N = (float)model->num_parameters;
    float flops = 6.0f * N * (float)num_tokens;
    float mfu = flops / (dt * get_flops_promised(deviceProp.name, deviceProp.major + deviceProp.minor * 0.1f));
    return mfu;
}

// ============================================================================
// main()
// ============================================================================

#include <sys/time.h>
static double time_in_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    // ---- Parse arguments ----
    const char *model_str    = "llama3.2:1b";
    const char *input_bin    = "dev/data/fineweb100B/fineweb_train_*.bin";
    const char *input_val_bin= "dev/data/fineweb100B/fineweb_val_*.bin";
    const char *output_dir   = "";
    int batch_size           = 4;
    int sequence_length      = 512;
    int total_batch_size     = 0;        // 0 = auto (B*T*ddp_world)
    int num_iterations       = -1;       // -1 = auto
    int inference_only       = 0;
    float learning_rate      = 3e-4f;
    int warmup_iters         = 0;
    float lr_decay_frac      = 1.0f;
    float weight_decay       = 0.0f;
    float grad_clip          = 1.0f;
    int val_loss_every       = 0;
    int val_max_steps        = 20;
    int overfit_single_batch = 0;
    int tensorcores          = 1;
    const char *device_str   = "";
    int zero_stage           = 1;
    int compile              = 0;
    const char *dtype_str    = "bfloat16";
    int test_tokenizer       = 0;

    for (int i = 1; i < argc; i++) {
#define PARSE_STR(flag, var)  if (strcmp(argv[i], flag) == 0) { var = argv[++i]; continue; }
#define PARSE_INT(flag, var)  if (strcmp(argv[i], flag) == 0) { var = atoi(argv[++i]); continue; }
#define PARSE_FLT(flag, var)  if (strcmp(argv[i], flag) == 0) { var = atof(argv[++i]); continue; }
        PARSE_STR("--model",                 model_str)
        PARSE_STR("--input_bin",             input_bin)
        PARSE_STR("--input_val_bin",         input_val_bin)
        PARSE_STR("--output_dir",            output_dir)
        PARSE_INT("--batch_size",            batch_size)
        PARSE_INT("--sequence_length",       sequence_length)
        PARSE_INT("--total_batch_size",      total_batch_size)
        PARSE_INT("--num_iterations",        num_iterations)
        PARSE_INT("--inference_only",        inference_only)
        PARSE_FLT("--learning_rate",         learning_rate)
        PARSE_INT("--warmup_iters",          warmup_iters)
        PARSE_FLT("--learning_rate_decay_frac", lr_decay_frac)
        PARSE_FLT("--weight_decay",          weight_decay)
        PARSE_FLT("--grad_clip",             grad_clip)
        PARSE_INT("--val_loss_every",        val_loss_every)
        PARSE_INT("--val_max_steps",         val_max_steps)
        PARSE_INT("--overfit_single_batch",  overfit_single_batch)
        PARSE_INT("--tensorcores",           tensorcores)
        PARSE_STR("--device",               device_str)
        PARSE_INT("--zero_stage",            zero_stage)
        PARSE_INT("--compile",               compile)
        PARSE_STR("--dtype",                dtype_str)
        PARSE_INT("--test_tokenizer",        test_tokenizer)
        fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(EXIT_FAILURE);
    }

    // SF16 tokenizer self-test (optional)
    if (test_tokenizer) { sf16_tokenizer_self_test(); return 0; }

    // ---- DDP setup ----
    multi_gpu_config = multi_gpu_config_init(-1, -1, -1, (char *)"", (char *)"", (char *)"");
    bool master_process = (multi_gpu_config.process_rank == 0);
    int  ddp_rank       = multi_gpu_config.process_rank;
    int  ddp_world_size = multi_gpu_config.num_processes;
    int  ddp_local_rank = multi_gpu_config.local_device_idx;
    (void)compile;

    // ---- Device ----
    char device_buf[64];
    if (strlen(device_str) > 0) {
        strncpy(device_buf, device_str, sizeof(device_buf)-1);
    } else {
        snprintf(device_buf, sizeof(device_buf), "cuda:%d", ddp_local_rank);
    }
    int device_id = ddp_local_rank;
    cudaCheck(cudaSetDevice(device_id));
    cudaCheck(cudaGetDeviceProperties(&deviceProp, device_id));
    cudaCheck(cudaStreamCreate(&main_stream));

    printf0("Device: %s | SM %d.%d\n", deviceProp.name,
            deviceProp.major, deviceProp.minor);
    printf0("Using model: %s | dtype: %s\n", model_str, dtype_str);
#if defined(ENABLE_Q115)
    printf0("SF16/Q1.15 mode enabled\n");
#endif

    if (tensorcores) cudaCheck(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 1));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cublasCheck(cublasCreate(&cublas_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision for FP32 mode, otherwise default compute type for BF16
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : cublas_compute;

    // ---- Batch size logic ----
    int B = batch_size;
    int T = sequence_length;
    assert(T >= 1 && T <= 8192);
    if (total_batch_size == 0)
        total_batch_size = B * T * ddp_world_size;
    int tokens_per_fwdbwd = B * T * ddp_world_size;
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    printf0("total batch size: %d | grad_accum_steps: %d\n",
            total_batch_size, grad_accum_steps);

    // ---- Logging ----
    char *logfile = nullptr;
    if (output_dir && strlen(output_dir) > 0) {
        create_dir_if_not_exists(output_dir);
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/main.log", output_dir);
        logfile = filename_buffer;
        FILE *lf = fopen(logfile, "w"); if (lf) fclose(lf);
    }

    // ---- Init model ----
    LLaMA32 model;
    llama32_init_common(&model);
    llama32_random_init(&model, model_str);

    // Set up ZeRO sharding
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);

    // ---- Data loaders ----
    DataLoader train_loader;
    dataloader_init(&train_loader, input_bin, B, T, ddp_rank, ddp_world_size, 1);
    
    if (num_iterations == -1) {
        num_iterations = train_loader.num_tokens / total_batch_size;
    }
    printf0("Training data: %lld tokens, %d steps\n", (long long)train_loader.num_tokens, num_iterations);

    DataLoader val_loader;
    bool has_val = (strlen(input_val_bin) > 0);
    if (has_val) dataloader_init(&val_loader, input_val_bin, B, T, ddp_rank, ddp_world_size, 0);

    // ---- Allocate training state ----
    llama32_allocate_state(&model, B, T);

    // ---- LR scheduler ----
    LearningRateScheduler lr_sched;
    lr_scheduler_init(&lr_sched, "cosine", learning_rate, warmup_iters, num_iterations, lr_decay_frac);

    // ---- Outlier detectors ----
    OutlierDetector loss_detector, grad_norm_detector;
    init_detector(&loss_detector);
    init_detector(&grad_norm_detector);

    // ---- cudaEvent timing ----
    cudaEvent_t ev_start, ev_end;
    cudaCheck(cudaEventCreate(&ev_start));
    cudaCheck(cudaEventCreate(&ev_end));

    // ---- Training loop ----
    printf0("starting training loop...\n"); fflush(stdout);
    float norm = -1.0f;
    for (int step = 0; step <= num_iterations; step++) {
        bool last_step = (step == num_iterations);

        // Validation
        if (has_val && val_loss_every > 0 &&
            (step % val_loss_every == 0 || last_step)) {
            model.mean_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int s = 0; s < val_max_steps; s++) {
                dataloader_next_batch(&val_loader);
                model.mean_loss += llama32_validate(&model,
                    val_loader.inputs, val_loader.targets, B, T);
            }
            model.mean_loss /= val_max_steps;
            float ppl = expf(model.mean_loss);
            printf0("val loss %.4f | ppl %.2f\n", model.mean_loss, ppl);
            if (master_process && logfile) {
                FILE *lf = fopen(logfile, "a");
                if (lf) { fprintf(lf, "s:%d tel:%f\n", step, model.mean_loss); fclose(lf); }
            }
        }

        if (last_step) break;

        // Training
        cudaCheck(cudaEventRecord(ev_start));
        for (int micro = 0; micro < grad_accum_steps; micro++) {
            if (overfit_single_batch) dataloader_reset(&train_loader);
            dataloader_next_batch(&train_loader);
            llama32_forward(&model, train_loader.inputs, B, T);
            llama32_backward_and_reduce(&model, train_loader.inputs,
                                        train_loader.targets, grad_accum_steps, micro);
        }

        norm = llama32_calculate_grad_norm(&model, &multi_gpu_config);
        float zloss = (float)update_detector(&loss_detector, (double)model.mean_loss);
        float zgrad = (float)update_detector(&grad_norm_detector, (double)norm);
        (void)zloss; (void)zgrad;

        float lr = get_learning_rate(&lr_sched, step);
        float grad_scale = (grad_clip > 0.0f && norm > grad_clip)
                           ? grad_clip / norm : 1.0f;
        llama32_update(&model, lr, 0.9f, 0.95f, 1e-8f, weight_decay,
                       grad_scale, step+1, &multi_gpu_config);

        cudaCheck(cudaEventRecord(ev_end));
        cudaCheck(cudaEventSynchronize(ev_end));
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, ev_start, ev_end));
        float tok_s = (float)(grad_accum_steps * ddp_world_size * B * T) / (time_elapsed_ms * 1e-3f);
        float mfu = llama32_estimate_mfu(&model, grad_accum_steps * B * T, time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %.6f | norm %.4f | lr %.2e | %.2f ms | %.0f tok/s | MFU %.2f%%\n",
                step+1, num_iterations, model.mean_loss, norm, lr,
                time_elapsed_ms, tok_s, mfu * 100.0f);

        if (master_process && logfile) {
            FILE *lf = fopen(logfile, "a");
            if (lf) { fprintf(lf, "s:%d trl:%f\n", step, model.mean_loss); fclose(lf); }
        }

        // Save checkpoint every 500 steps
        if (master_process && output_dir && strlen(output_dir) > 0
            && step > 0 && step % 500 == 0) {
            snprintf(filename_buffer, sizeof(filename_buffer),
                     "%s/llama32_%s_step%05d.bin", output_dir,
                     PRECISION_MODE == PRECISION_BF16 ? "bf16" : "q115", step);
            llama32_write_checkpoint(&model, filename_buffer);
        }
    }

    // ---- Cleanup ----
    multi_gpu_config_free(&multi_gpu_config);
    dataloader_free(&train_loader);
    if (has_val) dataloader_free(&val_loader);
    cudaCheck(cudaFree(model.params_memory));
    cudaCheck(cudaFree(model.grads_memory));
    cudaCheck(cudaFree(model.acts_memory));
    cudaCheck(cudaFree(model.d_freqs_cis));
    cudaCheck(cudaFree(cublaslt_workspace));
    cudaCheck(cudaStreamDestroy(main_stream));
    printf0("Training complete.\n");
    return 0;
}
