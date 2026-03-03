/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low
cost There will be other versions of this code that specialize it and make it
fast.

Modified to use Q1.15 fixed-point arithmetic for:
- Embeddings, transformer weights, activations, attention, FFN,
residual/layernorm, logits
- Gradients, loss, and optimizer remain in float32 for numerical stability
*/

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch,
// dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// Q1.15 Fixed-Point Arithmetic
// Q1.15 format: 1 sign bit, 15 fractional bits
// Range: [-1.0, 0.99997] with precision ~0.00003
// Represented as int16_t where -32768 = -1.0 and 32767 = 0.99997

typedef int16_t q115_t;

// Conversion constants
#define Q115_SCALE 32768.0f
#define Q115_MAX 32767
#define Q115_MIN -32768
#define Q115_OVERFLOW_THRESHOLD                                                \
  0.95f // Clamp values to prevent overflow in multiplications

// Convert float to Q1.15
static inline q115_t float_to_q115(float x) {
  // Clamp to prevent overflow
  if (x > Q115_OVERFLOW_THRESHOLD)
    x = Q115_OVERFLOW_THRESHOLD;
  if (x < -Q115_OVERFLOW_THRESHOLD)
    x = -Q115_OVERFLOW_THRESHOLD;

  float scaled = x * Q115_SCALE;
  int32_t rounded = (int32_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));

  // Additional safety clamp
  if (rounded > Q115_MAX)
    rounded = Q115_MAX;
  if (rounded < Q115_MIN)
    rounded = Q115_MIN;

  return (q115_t)rounded;
}

// Convert Q1.15 to float
static inline float q115_to_float(q115_t x) { return (float)x / Q115_SCALE; }

// Q1.15 multiplication with saturation
static inline q115_t q115_mul(q115_t a, q115_t b) {
  int32_t result = ((int32_t)a * (int32_t)b) >> 15;

  // Saturation
  if (result > Q115_MAX)
    return Q115_MAX;
  if (result < Q115_MIN)
    return Q115_MIN;

  return (q115_t)result;
}

// Q1.15 addition with saturation
static inline q115_t q115_add(q115_t a, q115_t b) {
  int32_t result = (int32_t)a + (int32_t)b;

  // Saturation
  if (result > Q115_MAX)
    return Q115_MAX;
  if (result < Q115_MIN)
    return Q115_MIN;

  return (q115_t)result;
}

// Bulk conversion functions with OpenMP parallelization
void float_array_to_q115(q115_t *dst, const float *src, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    dst[i] = float_to_q115(src[i]);
  }
}

void q115_array_to_float(float *dst, const q115_t *src, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    dst[i] = q115_to_float(src[i]);
  }
}

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(q115_t *out, int *inp, q115_t *wte, q115_t *wpe, int B,
                     int T, int C) {
// out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing
// token & position inp is (B,T) of integers, holding the token ids at each
// (b,t) position wte is (V,C) of token embeddings, short for "weight token
// embeddings" wpe is (maxT,C) of position embeddings, short for "weight
// positional embedding"
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the output position in out[b,t,:]
      q115_t *out_bt = out + b * T * C + t * C;
      // get the index of the token at inp[b, t]
      int ix = inp[b * T + t];
      // seek to the position in wte corresponding to the token
      q115_t *wte_ix = wte + ix * C;
      // seek to the position in wpe corresponding to the position
      q115_t *wpe_t = wpe + t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (int i = 0; i < C; i++) {
        out_bt[i] = q115_add(wte_ix[i], wpe_t[i]);
      }
    }
  }
}

void encoder_backward(float *dwte, float *dwpe, float *dout, int *inp, int B,
                      int T, int C) {
  // Gradients remain in float for numerical stability
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float *dwte_ix = dwte + ix * C;
      float *dwpe_t = dwpe + t * C;
      for (int i = 0; i < C; i++) {
        float d = dout_bt[i];
        dwte_ix[i] += d;
        dwpe_t[i] += d;
      }
    }
  }
}

void layernorm_forward(q115_t *out, float *mean, float *rstd, q115_t *inp,
                       q115_t *weight, q115_t *bias, int B, int T, int C) {
  // reference:
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html both inp
  // and out are (B,T,C) of the activations (in Q1.15) mean and rstd are (B,T)
  // buffers (in float), to be used later in backward pass at each position
  // (b,t) of the input, the C-dimensional vector of activations gets
  // normalized, then scaled and shifted
  float eps = 1e-5f;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      q115_t *x = inp + b * T * C + t * C;
      // calculate the mean (convert to float for computation)
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += q115_to_float(x[i]);
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = q115_to_float(x[i]) - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd (reciprocal standard deviation)
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      q115_t *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (q115_to_float(x[i]) - m)); // normalize
        float o = n * q115_to_float(weight[i]) +
                  q115_to_float(bias[i]); // scale and shift
        out_bt[i] = float_to_q115(o);     // convert back to Q1.15
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

void layernorm_backward(float *dinp, float *dweight, float *dbias, float *dout,
                        q115_t *inp, q115_t *weight, float *mean, float *rstd,
                        int B, int T, int C) {
  // Backward pass remains in float for numerical stability
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dout_bt = dout + b * T * C + t * C;
      q115_t *inp_bt = inp + b * T * C + t * C;
      float *dinp_bt = dinp + b * T * C + t * C;
      float mean_bt = mean[b * T + t];
      float rstd_bt = rstd[b * T + t];

      // first: two reduce operations
      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for (int i = 0; i < C; i++) {
        float norm_bti = (q115_to_float(inp_bt[i]) - mean_bt) * rstd_bt;
        float dnorm_i = q115_to_float(weight[i]) * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
      }
      dnorm_mean = dnorm_mean / C;
      dnorm_norm_mean = dnorm_norm_mean / C;

      // now iterate again and accumulate all the gradients
      for (int i = 0; i < C; i++) {
        float norm_bti = (q115_to_float(inp_bt[i]) - mean_bt) * rstd_bt;
        float dnorm_i = q115_to_float(weight[i]) * dout_bt[i];
        // gradient contribution to bias
        dbias[i] += dout_bt[i];
        // gradient contribution to weight
        dweight[i] += norm_bti * dout_bt[i];
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i;                    // term 1
        dval -= dnorm_mean;                 // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt;                    // final scale
        dinp_bt[i] += dval;
      }
    }
  }
}

void matmul_forward_naive(q115_t *out, const q115_t *inp, const q115_t *weight,
                          const q115_t *bias, int B, int T, int C, int OC) {
// the most naive implementation of matrix multiplication
// this serves as an algorithmic reference, and as a fallback for
// unfriendly input shapes inside matmul_forward(), below.
// Uses Q1.15 arithmetic with accumulation in float for precision
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      int bt = b * T + t;
      for (int o = 0; o < OC; o++) {
        float val = (bias != NULL) ? q115_to_float(bias[o]) : 0.0f;
        for (int i = 0; i < C; i++) {
          // Accumulate in float for better precision
          val +=
              q115_to_float(inp[bt * C + i]) * q115_to_float(weight[o * C + i]);
        }
        out[bt * OC + o] = float_to_q115(val);
      }
    }
  }
}

void matmul_forward(q115_t *out, const q115_t *inp, const q115_t *weight,
                    const q115_t *bias, int B, int T, int C, int OC) {
  // most of the running time is spent here and in matmul_backward
  // therefore, the implementation below is very mildly optimized
  // this function is otherwise identical to that of matmul_forward_naive()
  // OC is short for "output channels"
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
  // Uses Q1.15 arithmetic with accumulation in float for precision

  // make sure the tiled loop will be correct or fallback to naive version
  const int LOOP_UNROLL = 8;
  if (B * T % LOOP_UNROLL != 0) {
    matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
    return;
  }

// collapse the B and T loops into one and turn it into a strided loop.
// then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many
// times
#pragma omp parallel for
  for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
    // Pre-convert input slice to float to reduce conversion overhead
    // Use malloc instead of VLA for MSVC compatibility
    float *inp_cache = (float *)malloc(LOOP_UNROLL * C * sizeof(float));
    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
      int bt = obt + ibt;
      for (int i = 0; i < C; i++) {
        inp_cache[ibt * C + i] = q115_to_float(inp[bt * C + i]);
      }
    }

    for (int o = 0; o < OC; o++) {
      // we'll keep LOOP_UNROLL many results in float for precision
      float result[LOOP_UNROLL];
      // initialize the bias, if it exists
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
        result[ibt] = (bias != NULL) ? q115_to_float(bias[o]) : 0.0f;
      }
      // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
      // the value of weight[i + o * C] and reuse it.
      // we compile with -Ofast, so the compiler will turn the inner loop into
      // FMAs
      for (int i = 0; i < C; i++) {
        float w = q115_to_float(weight[i + o * C]);
        for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
          result[ibt] += inp_cache[ibt * C + i] * w;
        }
      }
      // write back results to main memory, converting back to Q1.15
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
        int bt = obt + ibt;
        out[bt * OC + o] = float_to_q115(result[ibt]);
      }
    }
    free(inp_cache);
  }
}

void matmul_backward(float *dinp, float *dweight, float *dbias,
                     const float *dout, const q115_t *inp, const q115_t *weight,
                     int B, int T, int C, int OC) {
// most of the running time is spent here and in matmul_forward
// this backward could be done in a single "round" of loops
// but that doesn't afford an efficient parallelization strategy
// Gradients remain in float for numerical stability

// backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *dout_bt = dout + b * T * OC + t * OC;
      float *dinp_bt = dinp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
        const q115_t *wrow = weight + o * C;
        float d = dout_bt[o];
        for (int i = 0; i < C; i++) {
          dinp_bt[i] += q115_to_float(wrow[i]) * d;
        }
      }
    }
  }
// backward into weight/bias, parallelize over output channels OC
#pragma omp parallel for
  for (int o = 0; o < OC; o++) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        const float *dout_bt = dout + b * T * OC + t * OC;
        const q115_t *inp_bt = inp + b * T * C + t * C;
        float *dwrow = dweight + o * C;
        float d = dout_bt[o];
        if (dbias != NULL) {
          dbias[o] += d;
        }
        for (int i = 0; i < C; i++) {
          dwrow[i] += q115_to_float(inp_bt[i]) * d;
        }
      }
    }
  }
}

void attention_forward(q115_t *out, float *preatt, float *att, q115_t *inp,
                       int B, int T, int C, int NH) {
  // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors (in
  // Q1.15) preatt, att are (B, NH, T, T) (in float for numerical stability in
  // softmax) that holds the pre-attention and post-attention scores (used in
  // backward) output is (B, T, C) (in Q1.15) attention is the only layer that
  // mixes information across time every other operation is applied at every
  // (b,t) position independently (and of course, no layer mixes information
  // across batch)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.0 / sqrtf(hs);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        q115_t *query_t = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
          q115_t *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

          // (query_t) dot (key_t2) - compute in float
          float val = 0.0f;
          for (int i = 0; i < hs; i++) {
            val += q115_to_float(query_t[i]) * q115_to_float(key_t2[i]);
          }
          val *= scale;
          if (val > maxval) {
            maxval = val;
          }

          preatt_bth[t2] = val;
        }

        // pass 2: calculate the exp and keep track of sum
        // maxval is being calculated and subtracted only for numerical
        // stability
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
          if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
          } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        q115_t *out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++) {
          out_bth[i] = 0;
        }
        for (int t2 = 0; t2 <= t; t2++) {
          q115_t *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                             C * 2; // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++) {
            // Accumulate in float then convert to Q1.15
            float acc = q115_to_float(out_bth[i]) +
                        att_btht2 * q115_to_float(value_t2[i]);
            out_bth[i] = float_to_q115(acc);
          }
        }
      }
    }
  }
}

void attention_backward(float *dinp, float *dpreatt, float *datt, float *dout,
                        q115_t *inp, float *att, int B, int T, int C, int NH) {
  // inp/dinp are (B, T, 3C) Q,K,V - inp in Q1.15, dinp in float
  // att/datt/dpreatt are (B, NH, T, T) in float
  // dout is (B, T, C) in float
  int C3 = C * 3;
  int hs = C / NH; // head size
  float scale = 1.f / sqrtf(hs);

  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
        float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
        float *dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
        q115_t *query_t = inp + b * T * C3 + t * C3 + h * hs;

        // backward pass 4, through the value accumulation
        float *dout_bth = dout + b * T * C + t * C + h * hs;
        for (int t2 = 0; t2 <= t; t2++) {
          q115_t *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                             C * 2; // +C*2 because it's value
          float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // out_bth[i] += att_bth[t2] * value_t2[i];
            // so now we have:
            datt_bth[t2] += q115_to_float(value_t2[i]) * dout_bth[i];
            dvalue_t2[i] += att_bth[t2] * dout_bth[i];
          }
        }

        // backward pass 2 & 3, the softmax
        // note that softmax (like e.g. tanh) doesn't need the input (preatt) to
        // backward
        for (int t2 = 0; t2 <= t; t2++) {
          for (int t3 = 0; t3 <= t; t3++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            dpreatt_bth[t3] += local_derivative * datt_bth[t2];
          }
        }

        // backward pass 1, the query @ key matmul
        for (int t2 = 0; t2 <= t; t2++) {
          q115_t *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          float *dkey_t2 =
              dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
            // so now we have:
            dquery_t[i] += q115_to_float(key_t2[i]) * dpreatt_bth[t2] * scale;
            dkey_t2[i] += q115_to_float(query_t[i]) * dpreatt_bth[t2] * scale;
          }
        }
      }
    }
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(q115_t *out, q115_t *inp, int N) {
// (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
// Compute in float for accuracy, convert back to Q1.15
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float x = q115_to_float(inp[i]);
    float cube = 0.044715f * x * x * x;
    float result = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    out[i] = float_to_q115(result);
  }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this
// flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, q115_t* inp, float* dout, int N) {
  // Backward pass remains in float
  for (int i = 0; i < N; i++) {
    float x = q115_to_float(inp[i]);
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                       (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] += local_grad * dout[i];
  }
}
#pragma float_control(pop)

void residual_forward(q115_t *out, q115_t *inp1, q115_t *inp2, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    out[i] = q115_add(inp1[i], inp2[i]);
  }
}

void residual_backward(float *dinp1, float *dinp2, float *dout, int N) {
  // Backward pass remains in float
  for (int i = 0; i < N; i++) {
    dinp1[i] += dout[i];
    dinp2[i] += dout[i];
  }
}

void softmax_forward(float *probs, q115_t *logits, int B, int T, int V,
                     int Vp) {
// output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t
// position) in float input: logits is (B,T,Vp) of the unnormalized log
// probabilities in Q1.15 Vp is the padded vocab size (for efficiency), V is the
// "real" vocab size example: Vp is 50304 and V is 50257 We compute softmax in
// float for numerical stability
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // probs <- softmax(logits)
      q115_t *logits_bt = logits + b * T * Vp + t * Vp;
      float *probs_bt = probs + b * T * Vp + t * Vp;

      // maxval is only calculated and subtracted for numerical stability
      float maxval = -10000.0f; // TODO something better
      for (int i = 0; i < V; i++) {
        float logit_val = q115_to_float(logits_bt[i]);
        if (logit_val > maxval) {
          maxval = logit_val;
        }
      }
      float sum = 0.0f;
      for (int i = 0; i < V; i++) {
        probs_bt[i] = expf(q115_to_float(logits_bt[i]) - maxval);
        sum += probs_bt[i];
      }
      // note we only loop to V, leaving the padded dimensions
      for (int i = 0; i < V; i++) {
        probs_bt[i] /= sum;
      }
      // for extra super safety we may wish to include this too,
      // forcing the probabilities here to be zero, but it shouldn't matter
      for (int i = V; i < Vp; i++) {
        probs_bt[i] = 0.0f;
      }
    }
  }
}

void crossentropy_forward(float *losses, float *probs, int *targets, int B,
                          int T, int Vp) {
  // output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,Vp) of the probabilities
  // input: targets is (B,T) of integers giving the correct index in logits
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // loss = -log(probs[target])
      float *probs_bt = probs + b * T * Vp + t * Vp;
      int ix = targets[b * T + t];
      losses[b * T + t] = -logf(probs_bt[ix]);
    }
  }
}

void crossentropy_softmax_backward(float *dlogits, float *dlosses, float *probs,
                                   int *targets, int B, int T, int V, int Vp) {
  // backwards through both softmax and crossentropy
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float *dlogits_bt = dlogits + b * T * Vp + t * Vp;
      float *probs_bt = probs + b * T * Vp + t * Vp;
      float dloss = dlosses[b * T + t];
      int ix = targets[b * T + t];
      // note we only loop to V, leaving the padded dimensions
      // of dlogits untouched, so gradient there stays at zero
      for (int i = 0; i < V; i++) {
        float p = probs_bt[i];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] += (p - indicator) * dloss;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
  int max_seq_len;       // max sequence length, e.g. 1024
  int vocab_size;        // vocab size, e.g. 50257
  int padded_vocab_size; // padded to e.g. %128==0, 50304
  int num_layers;        // number of layers, e.g. 12
  int num_heads;         // number of heads in attention, e.g. 12
  int channels;          // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model (in Q1.15 fixed-point)
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  q115_t *wte;      // (V, C)
  q115_t *wpe;      // (maxT, C)
  q115_t *ln1w;     // (L, C)
  q115_t *ln1b;     // (L, C)
  q115_t *qkvw;     // (L, 3*C, C)
  q115_t *qkvb;     // (L, 3*C)
  q115_t *attprojw; // (L, C, C)
  q115_t *attprojb; // (L, C)
  q115_t *ln2w;     // (L, C)
  q115_t *ln2b;     // (L, C)
  q115_t *fcw;      // (L, 4*C, C)
  q115_t *fcb;      // (L, 4*C)
  q115_t *fcprojw;  // (L, C, 4*C)
  q115_t *fcprojb;  // (L, C)
  q115_t *lnfw;     // (C)
  q115_t *lnfb;     // (C)
} ParameterTensors;

// Gradient tensors (always in float for numerical stability)
typedef struct {
  float *wte;      // (V, C)
  float *wpe;      // (maxT, C)
  float *ln1w;     // (L, C)
  float *ln1b;     // (L, C)
  float *qkvw;     // (L, 3*C, C)
  float *qkvb;     // (L, 3*C)
  float *attprojw; // (L, C, C)
  float *attprojb; // (L, C)
  float *ln2w;     // (L, C)
  float *ln2b;     // (L, C)
  float *fcw;      // (L, 4*C, C)
  float *fcb;      // (L, 4*C)
  float *fcprojw;  // (L, C, 4*C)
  float *fcprojb;  // (L, C)
  float *lnfw;     // (C)
  float *lnfb;     // (C)
} ParameterTensorsGrad;

void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
  size_t Vp = config.padded_vocab_size;
  size_t C = config.channels;
  size_t maxT = config.max_seq_len;
  size_t L = config.num_layers;
  param_sizes[0] = Vp * C;           // wte
  param_sizes[1] = maxT * C;         // wpe
  param_sizes[2] = L * C;            // ln1w
  param_sizes[3] = L * C;            // ln1b
  param_sizes[4] = L * (3 * C) * C;  // qkvw
  param_sizes[5] = L * (3 * C);      // qkvb
  param_sizes[6] = L * C * C;        // attprojw
  param_sizes[7] = L * C;            // attprojb
  param_sizes[8] = L * C;            // ln2w
  param_sizes[9] = L * C;            // ln2b
  param_sizes[10] = L * (4 * C) * C; // fcw
  param_sizes[11] = L * (4 * C);     // fcb
  param_sizes[12] = L * C * (4 * C); // fcprojw
  param_sizes[13] = L * C;           // fcprojb
  param_sizes[14] = C;               // lnfw
  param_sizes[15] = C;               // lnfb
}

// allocate memory for the parameters and point the individual tensors to the
// right places
q115_t *malloc_and_point_parameters(ParameterTensors *params,
                                    size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once (Q1.15 uses int16_t)
  q115_t *params_memory =
      (q115_t *)mallocCheck(num_parameters * sizeof(q115_t));
  // assign all the tensors
  q115_t **ptrs[] = {
      &params->wte,     &params->wpe,     &params->ln1w,     &params->ln1b,
      &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
      &params->ln2w,    &params->ln2b,    &params->fcw,      &params->fcb,
      &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
  q115_t *params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}

// allocate memory for the gradient parameters (always float)
float *malloc_and_point_gradients(ParameterTensorsGrad *grads,
                                  size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all gradients all at once (float)
  float *grads_memory = (float *)mallocCheck(num_parameters * sizeof(float));
  // assign all the tensors
  float **ptrs[] = {
      &grads->wte,     &grads->wpe,     &grads->ln1w,     &grads->ln1b,
      &grads->qkvw,    &grads->qkvb,    &grads->attprojw, &grads->attprojb,
      &grads->ln2w,    &grads->ln2b,    &grads->fcw,      &grads->fcb,
      &grads->fcprojw, &grads->fcprojb, &grads->lnfw,     &grads->lnfb};
  float *grads_memory_iterator = grads_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = grads_memory_iterator;
    grads_memory_iterator += param_sizes[i];
  }
  return grads_memory;
}

// Helper function to convert bfloat16 to float32
float bf16_to_fp32(uint16_t bf16) {
  // bfloat16 is stored as the upper 16 bits of a float32
  uint32_t fp32_bits = ((uint32_t)bf16) << 16;
  float result;
  memcpy(&result, &fp32_bits, sizeof(float));
  return result;
}

// Helper function to read parameters from file, handling both fp32 and bf16
// Convert to Q1.15 format
void read_parameters(q115_t *params_memory, size_t num_parameters, FILE *file,
                     int version) {
  // Read parameters as float first, then convert to Q1.15
  float *float_buffer = (float *)mallocCheck(num_parameters * sizeof(float));

  if (version == 3) {
    // float32 - direct read
    freadCheck(float_buffer, sizeof(float), num_parameters, file);
  } else if (version == 5) {
    // bfloat16 - read as uint16 and convert to float32
    uint16_t *bf16_buffer =
        (uint16_t *)mallocCheck(num_parameters * sizeof(uint16_t));
    freadCheck(bf16_buffer, sizeof(uint16_t), num_parameters, file);
    // convert bfloat16 to float32
    for (size_t i = 0; i < num_parameters; i++) {
      float_buffer[i] = bf16_to_fp32(bf16_buffer[i]);
    }
    free(bf16_buffer);
  }

  // Convert float to Q1.15 with scaled initialization closer to zero
  // Scale down weights by a factor to keep them in a safer range for Q1.15
  float init_scale = 0.5f; // Scale down to prevent overflow in early training
  for (size_t i = 0; i < num_parameters; i++) {
    params_memory[i] = float_to_q115(float_buffer[i] * init_scale);
  }

  free(float_buffer);
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
  q115_t *encoded;   // (B, T, C) - Q1.15
  q115_t *ln1;       // (L, B, T, C) - Q1.15
  float *ln1_mean;   // (L, B, T) - float for stability
  float *ln1_rstd;   // (L, B, T) - float for stability
  q115_t *qkv;       // (L, B, T, 3*C) - Q1.15
  q115_t *atty;      // (L, B, T, C) - Q1.15
  float *preatt;     // (L, B, NH, T, T) - float for stability
  float *att;        // (L, B, NH, T, T) - float for stability
  q115_t *attproj;   // (L, B, T, C) - Q1.15
  q115_t *residual2; // (L, B, T, C) - Q1.15
  q115_t *ln2;       // (L, B, T, C) - Q1.15
  float *ln2_mean;   // (L, B, T) - float for stability
  float *ln2_rstd;   // (L, B, T) - float for stability
  q115_t *fch;       // (L, B, T, 4*C) - Q1.15
  q115_t *fch_gelu;  // (L, B, T, 4*C) - Q1.15
  q115_t *fcproj;    // (L, B, T, C) - Q1.15
  q115_t *residual3; // (L, B, T, C) - Q1.15
  q115_t *lnf;       // (B, T, C) - Q1.15
  float *lnf_mean;   // (B, T) - float for stability
  float *lnf_rstd;   // (B, T) - float for stability
  q115_t *logits;    // (B, T, V) - Q1.15
  float *probs;      // (B, T, V) - float for softmax
  float *losses;     // (B, T) - float
} ActivationTensors;

// Activation gradients (always in float for numerical stability)
typedef struct {
  float *encoded;   // (B, T, C)
  float *ln1;       // (L, B, T, C)
  float *ln1_mean;  // (L, B, T) - not used
  float *ln1_rstd;  // (L, B, T) - not used
  float *qkv;       // (L, B, T, 3*C)
  float *atty;      // (L, B, T, C)
  float *preatt;    // (L, B, NH, T, T)
  float *att;       // (L, B, NH, T, T)
  float *attproj;   // (L, B, T, C)
  float *residual2; // (L, B, T, C)
  float *ln2;       // (L, B, T, C)
  float *ln2_mean;  // (L, B, T) - not used
  float *ln2_rstd;  // (L, B, T) - not used
  float *fch;       // (L, B, T, 4*C)
  float *fch_gelu;  // (L, B, T, 4*C)
  float *fcproj;    // (L, B, T, C)
  float *residual3; // (L, B, T, C)
  float *lnf;       // (B, T, C)
  float *lnf_mean;  // (B, T) - not used
  float *lnf_rstd;  // (B, T) - not used
  float *logits;    // (B, T, V)
  float *probs;     // (B, T, V)
  float *losses;    // (B, T)
} ActivationTensorsGrad;

void fill_in_activation_sizes(size_t *act_sizes, size_t *act_sizes_bytes,
                              GPT2Config config, int B, int T) {
  // Calculate sizes in terms of number of elements
  // Some activations are Q1.15 (2 bytes), others are float (4 bytes)
  size_t C = config.channels;
  size_t NH = config.num_heads;
  size_t L = config.num_layers;
  size_t Vp = config.padded_vocab_size;

  // Sizes in elements
  act_sizes[0] = B * T * C;          // encoded - Q1.15
  act_sizes[1] = L * B * T * C;      // ln1 - Q1.15
  act_sizes[2] = L * B * T;          // ln1_mean - float
  act_sizes[3] = L * B * T;          // ln1_rstd - float
  act_sizes[4] = L * B * T * 3 * C;  // qkv - Q1.15
  act_sizes[5] = L * B * T * C;      // atty - Q1.15
  act_sizes[6] = L * B * NH * T * T; // preatt - float
  act_sizes[7] = L * B * NH * T * T; // att - float
  act_sizes[8] = L * B * T * C;      // attproj - Q1.15
  act_sizes[9] = L * B * T * C;      // residual2 - Q1.15
  act_sizes[10] = L * B * T * C;     // ln2 - Q1.15
  act_sizes[11] = L * B * T;         // ln2_mean - float
  act_sizes[12] = L * B * T;         // ln2_rstd - float
  act_sizes[13] = L * B * T * 4 * C; // fch - Q1.15
  act_sizes[14] = L * B * T * 4 * C; // fch_gelu - Q1.15
  act_sizes[15] = L * B * T * C;     // fcproj - Q1.15
  act_sizes[16] = L * B * T * C;     // residual3 - Q1.15
  act_sizes[17] = B * T * C;         // lnf - Q1.15
  act_sizes[18] = B * T;             // lnf_mean - float
  act_sizes[19] = B * T;             // lnf_rstd - float
  act_sizes[20] = B * T * Vp;        // logits - Q1.15
  act_sizes[21] = B * T * Vp;        // probs - float
  act_sizes[22] = B * T;             // losses - float

  // Sizes in bytes (Q1.15 = 2 bytes, float = 4 bytes)
  int is_q115[] = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                   0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0};
  for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    act_sizes_bytes[i] =
        act_sizes[i] * (is_q115[i] ? sizeof(q115_t) : sizeof(float));
  }
}

void *malloc_and_point_activations(ActivationTensors *acts,
                                   size_t *act_sizes_bytes) {
  // Calculate total bytes needed (mixed Q1.15 and float)
  size_t total_bytes = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    total_bytes += act_sizes_bytes[i];
  }
  // Allocate memory as bytes
  char *acts_memory = (char *)mallocCheck(total_bytes);

  // Assign pointers with appropriate casting
  char *ptr = acts_memory;
  acts->encoded = (q115_t *)ptr;
  ptr += act_sizes_bytes[0];
  acts->ln1 = (q115_t *)ptr;
  ptr += act_sizes_bytes[1];
  acts->ln1_mean = (float *)ptr;
  ptr += act_sizes_bytes[2];
  acts->ln1_rstd = (float *)ptr;
  ptr += act_sizes_bytes[3];
  acts->qkv = (q115_t *)ptr;
  ptr += act_sizes_bytes[4];
  acts->atty = (q115_t *)ptr;
  ptr += act_sizes_bytes[5];
  acts->preatt = (float *)ptr;
  ptr += act_sizes_bytes[6];
  acts->att = (float *)ptr;
  ptr += act_sizes_bytes[7];
  acts->attproj = (q115_t *)ptr;
  ptr += act_sizes_bytes[8];
  acts->residual2 = (q115_t *)ptr;
  ptr += act_sizes_bytes[9];
  acts->ln2 = (q115_t *)ptr;
  ptr += act_sizes_bytes[10];
  acts->ln2_mean = (float *)ptr;
  ptr += act_sizes_bytes[11];
  acts->ln2_rstd = (float *)ptr;
  ptr += act_sizes_bytes[12];
  acts->fch = (q115_t *)ptr;
  ptr += act_sizes_bytes[13];
  acts->fch_gelu = (q115_t *)ptr;
  ptr += act_sizes_bytes[14];
  acts->fcproj = (q115_t *)ptr;
  ptr += act_sizes_bytes[15];
  acts->residual3 = (q115_t *)ptr;
  ptr += act_sizes_bytes[16];
  acts->lnf = (q115_t *)ptr;
  ptr += act_sizes_bytes[17];
  acts->lnf_mean = (float *)ptr;
  ptr += act_sizes_bytes[18];
  acts->lnf_rstd = (float *)ptr;
  ptr += act_sizes_bytes[19];
  acts->logits = (q115_t *)ptr;
  ptr += act_sizes_bytes[20];
  acts->probs = (float *)ptr;
  ptr += act_sizes_bytes[21];
  acts->losses = (float *)ptr;
  ptr += act_sizes_bytes[22];

  return acts_memory;
}

float *malloc_and_point_activation_grads(ActivationTensorsGrad *acts,
                                         size_t *act_sizes) {
  // All activation gradients are float
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory = (float *)mallocCheck(num_activations * sizeof(float));
  float **ptrs[] = {
      &acts->encoded,   &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd,
      &acts->qkv,       &acts->atty,      &acts->preatt,   &acts->att,
      &acts->attproj,   &acts->residual2, &acts->ln2,      &acts->ln2_mean,
      &acts->ln2_rstd,  &acts->fch,       &acts->fch_gelu, &acts->fcproj,
      &acts->residual3, &acts->lnf,       &acts->lnf_mean, &acts->lnf_rstd,
      &acts->logits,    &acts->probs,     &acts->losses};
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

typedef struct {
  GPT2Config config;
  // the weights (parameters) of the model, and their sizes (Q1.15)
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  q115_t *params_memory;
  size_t num_parameters;
  // gradients of the weights (float for numerical stability)
  ParameterTensorsGrad grads;
  float *grads_memory;
  // buffers for the AdamW optimizer (float)
  float *m_memory;
  float *v_memory;
  // the activations of the model, and their sizes (mixed Q1.15 and float)
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];       // sizes in elements
  size_t act_sizes_bytes[NUM_ACTIVATION_TENSORS]; // sizes in bytes
  void *acts_memory;
  size_t num_activations; // total number of bytes
  // gradients of the activations (float for numerical stability)
  ActivationTensorsGrad grads_acts;
  float *grads_acts_memory;
  // other run state configuration
  int batch_size;  // the batch size (B) of current forward pass
  int seq_len;     // the sequence length (T) of current forward pass
  int *inputs;     // the input tokens for the current forward pass
  int *targets;    // the target tokens for the current forward pass
  float mean_loss; // after a forward pass with targets, will be populated with
                   // the mean loss
} GPT2;

void gpt2_init_random(GPT2 *model, int max_seq_len, int vocab_size,
                      int num_layers, int num_heads, int channels) {
  // Initialize model with random weights when checkpoint is not available
  // This creates a GPT-2 model from scratch with the given hyperparameters

  // Set up the configuration
  model->config.max_seq_len = max_seq_len;
  model->config.vocab_size = vocab_size;
  model->config.num_layers = num_layers;
  model->config.num_heads = num_heads;
  model->config.channels = channels;
  // Pad vocab size to be a multiple of 128 for efficiency
  model->config.padded_vocab_size =
      (vocab_size % 128 == 0) ? vocab_size : ((vocab_size / 128 + 1) * 128);

  printf("[GPT-2] Initializing from scratch with random weights\n");
  printf("max_seq_len: %d\n", max_seq_len);
  printf("vocab_size: %d\n", vocab_size);
  printf("padded_vocab_size: %d\n", model->config.padded_vocab_size);
  printf("num_layers: %d\n", num_layers);
  printf("num_heads: %d\n", num_heads);
  printf("channels: %d\n", channels);

  // allocate space for all the parameters
  fill_in_parameter_sizes(model->param_sizes, model->config);

  // count the number of parameters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("num_parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;

  // allocate memory for parameters and initialize with random values
  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes);

  // Initialize with small random values using a simple random number generator
  // For Q1.15: scale initialization to prevent overflow, values closer to zero
  // Using scaled initialization: range approximately [-0.1, 0.1] to be safe
  uint64_t seed = 12345;
  float init_scale = 0.1f; // Conservative scale for Q1.15
  for (size_t i = 0; i < num_parameters; i++) {
    // Simple random initialization
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    float random_val =
        (((seed * 0x2545F4914F6CDD1Dull) >> 32) / (float)UINT32_MAX) * 2.0f -
        1.0f;
    random_val *= init_scale; // Scale to [-0.1, 0.1]
    model->params_memory[i] = float_to_q115(random_val);
  }

  // other inits
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
  // Check if checkpoint file exists
  if (access(checkpoint_path, F_OK) == -1) {
    printf("Checkpoint file not found: %s\n", checkpoint_path);
    printf("Falling back to random initialization with default GPT-2 124M "
           "hyperparameters\n");
    // Use GPT-2 124M default hyperparameters
    gpt2_init_random(model, 1024, 50257, 12, 12, 768);
    return;
  }

  // read in model from a checkpoint file
  FILE *model_file = fopen(checkpoint_path, "rb");
  if (model_file == NULL) {
    printf("Error opening checkpoint file: %s\n", checkpoint_path);
    printf("Falling back to random initialization with default GPT-2 124M "
           "hyperparameters\n");
    gpt2_init_random(model, 1024, 50257, 12, 12, 768);
    return;
  }

  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    printf("Bad magic model file\n");
    fcloseCheck(model_file);
    printf("Falling back to random initialization\n");
    gpt2_init_random(model, 1024, 50257, 12, 12, 768);
    return;
  }

  // Support both version 3 (float32) and version 5 (bfloat16)
  int version = model_header[1];
  if (version != 3 && version != 5) {
    printf("Bad version in model file: %d\n", version);
    printf("---> HINT: try to re-run `python train_gpt2.py`\n");
    printf("Supported versions: 3 (float32), 5 (bfloat16)\n");
    fcloseCheck(model_file);
    printf("Falling back to random initialization\n");
    gpt2_init_random(model, 1024, 50257, 12, 12, 768);
    return;
  }

  printf("[GPT-2] Loading checkpoint: %s (version %d - %s)\n", checkpoint_path,
         version, version == 3 ? "float32" : "bfloat16");

  // read in hyperparameters
  size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];
  model->config.padded_vocab_size = Vp = model_header[7];
  printf("[GPT-2]\n");
  printf("max_seq_len: %zu\n", maxT);
  printf("vocab_size: %zu\n", V);
  printf("padded_vocab_size: %zu\n", Vp);
  printf("num_layers: %zu\n", L);
  printf("num_heads: %zu\n", NH);
  printf("channels: %zu\n", C);

  // allocate space for all the parameters and read them in
  fill_in_parameter_sizes(model->param_sizes, model->config);

  // count the number of parameters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("num_parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;

  // allocate memory and read parameters (handling both fp32 and bf16)
  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes);
  read_parameters(model->params_memory, num_parameters, model_file, version);
  fcloseCheck(model_file);

  // other inits
  model->acts_memory = NULL;
  model->grads_memory = NULL;
  model->m_memory = NULL;
  model->v_memory = NULL;
  model->grads_acts_memory = NULL;
  model->inputs = NULL;
  model->targets = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
  model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, size_t B, size_t T) {
  // targets are optional and could be NULL

  // ensure the model was initialized or error out
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(1);
  }

  // convenience parameters (size_t to help prevent int overflow)
  size_t V = model->config.vocab_size;
  size_t Vp = model->config.padded_vocab_size;
  size_t L = model->config.num_layers;
  size_t NH = model->config.num_heads;
  size_t C = model->config.channels;

  // validate inputs, all indices must be in the range [0, V)
  for (int i = 0; i < B * T; i++) {
    assert(0 <= inputs[i] && inputs[i] < V);
    if (targets != NULL) {
      assert(0 <= targets[i] && targets[i] < V);
    }
  }

  // allocate space for all the activations if needed (done here, lazily)
  if (model->acts_memory == NULL) {
    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    fill_in_activation_sizes(model->act_sizes, model->act_sizes_bytes,
                             model->config, B, T);
    size_t num_activations_bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations_bytes += model->act_sizes_bytes[i];
    }
    printf("num_activations_bytes: %zu\n", num_activations_bytes);
    model->num_activations = num_activations_bytes;
    model->acts_memory =
        malloc_and_point_activations(&model->acts, model->act_sizes_bytes);
    // also create memory for caching inputs and targets
    model->inputs = (int *)mallocCheck(B * T * sizeof(int));
    model->targets = (int *)mallocCheck(
        B * T *
        sizeof(int)); // might be unused if we never have targets but it's small
  } else {
    // validate B,T is consistent with how we've allocated the memory before
    // in principle we could get more clever here in the future, for now this is
    // safest
    if (B != model->batch_size || T != model->seq_len) {
      printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size,
             model->seq_len, (int)B, (int)T);
      exit(EXIT_FAILURE);
    }
  }

  // cache the inputs/targets
  memcpy(model->inputs, inputs, B * T * sizeof(int));
  if (targets != NULL) {
    memcpy(model->targets, targets, B * T * sizeof(int));
  }

  // forward pass
  ParameterTensors params = model->params; // for brevity
  ActivationTensors acts = model->acts;
  q115_t *residual;
  encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T,
                  C); // encoding goes into residual[0]
  for (int l = 0; l < L; l++) {

    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

    // get the pointers of the weights for this layer (all Q1.15)
    q115_t *l_ln1w = params.ln1w + l * C;
    q115_t *l_ln1b = params.ln1b + l * C;
    q115_t *l_qkvw = params.qkvw + l * 3 * C * C;
    q115_t *l_qkvb = params.qkvb + l * 3 * C;
    q115_t *l_attprojw = params.attprojw + l * C * C;
    q115_t *l_attprojb = params.attprojb + l * C;
    q115_t *l_ln2w = params.ln2w + l * C;
    q115_t *l_ln2b = params.ln2b + l * C;
    q115_t *l_fcw = params.fcw + l * 4 * C * C;
    q115_t *l_fcb = params.fcb + l * 4 * C;
    q115_t *l_fcprojw = params.fcprojw + l * C * 4 * C;
    q115_t *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of the activations for this layer (mostly Q1.15, some
    // float)
    q115_t *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    q115_t *l_qkv = acts.qkv + l * B * T * 3 * C;
    q115_t *l_atty = acts.atty + l * B * T * C;
    float *l_preatt = acts.preatt + l * B * NH * T * T;
    float *l_att = acts.att + l * B * NH * T * T;
    q115_t *l_attproj = acts.attproj + l * B * T * C;
    q115_t *l_residual2 = acts.residual2 + l * B * T * C;
    q115_t *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    q115_t *l_fch = acts.fch + l * B * T * 4 * C;
    q115_t *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    q115_t *l_fcproj = acts.fcproj + l * B * T * C;
    q115_t *l_residual3 = acts.residual3 + l * B * T * C;

    // now do the forward pass
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b,
                      B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B * T * C);
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w,
                      l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
  }
  residual =
      acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual,
                    params.lnfw, params.lnfb, B, T, C);
  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
  softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

  // also forward the cross-entropy loss function if we have the targets
  if (targets != NULL) {
    crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T,
                         Vp);
    // for convenience also evaluate the mean loss
    float mean_loss = 0.0f;
    for (int i = 0; i < B * T; i++) {
      mean_loss += model->acts.losses[i];
    }
    mean_loss /= B * T;
    model->mean_loss = mean_loss;
  } else {
    // if we don't have targets, we don't have a loss
    model->mean_loss = -1.0f;
  }
}

void gpt2_zero_grad(GPT2 *model) {
  if (model->grads_memory != NULL) {
    memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
  }
  if (model->grads_acts_memory != NULL) {
    // Calculate total activation gradients size (all float)
    size_t total_act_grads = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      total_act_grads += model->act_sizes[i];
    }
    memset(model->grads_acts_memory, 0, total_act_grads * sizeof(float));
  }
}

void gpt2_backward(GPT2 *model) {

  // double check we forwarded previously, with targets
  if (model->mean_loss == -1.0f) {
    printf("Error: must forward with targets before backward\n");
    exit(1);
  }

  // lazily allocate the memory for gradients of the weights and activations, if
  // needed
  if (model->grads_memory == NULL) {
    model->grads_memory =
        malloc_and_point_gradients(&model->grads, model->param_sizes);
    model->grads_acts_memory =
        malloc_and_point_activation_grads(&model->grads_acts, model->act_sizes);
    gpt2_zero_grad(model);
  }

  // convenience shortcuts (and size_t to help prevent int overflow)
  size_t B = model->batch_size;
  size_t T = model->seq_len;
  size_t V = model->config.vocab_size;
  size_t Vp = model->config.padded_vocab_size;
  size_t L = model->config.num_layers;
  size_t NH = model->config.num_heads;
  size_t C = model->config.channels;

  // backward pass: go in the reverse order of the forward pass, and call
  // backward() functions
  ParameterTensors params = model->params; // for brevity
  ParameterTensorsGrad grads = model->grads;
  ActivationTensors acts = model->acts;
  ActivationTensorsGrad grads_acts = model->grads_acts;

  // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
  // technically this is a small, inline backward() pass of calculating
  // total, final loss as the mean over all losses over all (B,T) positions in
  // the batch
  float dloss_mean = 1.0f / (B * T);
  for (int i = 0; i < B * T; i++) {
    grads_acts.losses[i] = dloss_mean;
  }

  crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses,
                                acts.probs, model->targets, B, T, V, Vp);
  matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf,
                  params.wte, B, T, C, Vp);
  q115_t *residual =
      acts.residual3 + (L - 1) * B * T * C; // last layer's residual
  float *dresidual = grads_acts.residual3 +
                     (L - 1) * B * T * C; // write to last layer's residual
  layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf,
                     residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T,
                     C);

  for (int l = L - 1; l >= 0; l--) {

    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
    dresidual = l == 0 ? grads_acts.encoded
                       : grads_acts.residual3 + (l - 1) * B * T * C;

    // get the pointers of the weights for this layer (Q1.15)
    q115_t *l_ln1w = params.ln1w + l * C;
    q115_t *l_qkvw = params.qkvw + l * 3 * C * C;
    q115_t *l_attprojw = params.attprojw + l * C * C;
    q115_t *l_ln2w = params.ln2w + l * C;
    q115_t *l_fcw = params.fcw + l * 4 * C * C;
    q115_t *l_fcprojw = params.fcprojw + l * C * 4 * C;
    // get the pointers of the gradients of the weights for this layer
    float *dl_ln1w = grads.ln1w + l * C;
    float *dl_ln1b = grads.ln1b + l * C;
    float *dl_qkvw = grads.qkvw + l * 3 * C * C;
    float *dl_qkvb = grads.qkvb + l * 3 * C;
    float *dl_attprojw = grads.attprojw + l * C * C;
    float *dl_attprojb = grads.attprojb + l * C;
    float *dl_ln2w = grads.ln2w + l * C;
    float *dl_ln2b = grads.ln2b + l * C;
    float *dl_fcw = grads.fcw + l * 4 * C * C;
    float *dl_fcb = grads.fcb + l * 4 * C;
    float *dl_fcprojw = grads.fcprojw + l * C * 4 * C;
    float *dl_fcprojb = grads.fcprojb + l * C;
    // get the pointers of the activations for this layer (Q1.15)
    q115_t *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
    q115_t *l_qkv = acts.qkv + l * B * T * 3 * C;
    q115_t *l_atty = acts.atty + l * B * T * C;
    float *l_att = acts.att + l * B * NH * T * T;
    q115_t *l_residual2 = acts.residual2 + l * B * T * C;
    q115_t *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
    q115_t *l_fch = acts.fch + l * B * T * 4 * C;
    q115_t *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    // get the pointers of the gradients of the activations for this layer (all
    // float)
    float *dl_ln1 = grads_acts.ln1 + l * B * T * C;
    float *dl_qkv = grads_acts.qkv + l * B * T * 3 * C;
    float *dl_atty = grads_acts.atty + l * B * T * C;
    float *dl_preatt = grads_acts.preatt + l * B * NH * T * T;
    float *dl_att = grads_acts.att + l * B * NH * T * T;
    float *dl_attproj = grads_acts.attproj + l * B * T * C;
    float *dl_residual2 = grads_acts.residual2 + l * B * T * C;
    float *dl_ln2 = grads_acts.ln2 + l * B * T * C;
    float *dl_fch = grads_acts.fch + l * B * T * 4 * C;
    float *dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C;
    float *dl_fcproj = grads_acts.fcproj + l * B * T * C;
    float *dl_residual3 = grads_acts.residual3 + l * B * T * C;

    // backprop this layer
    residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
    matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu,
                    l_fcprojw, B, T, 4 * C, C);
    gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
    matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C,
                    4 * C);
    layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2,
                       l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
    residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
    matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty,
                    l_attprojw, B, T, C, C);
    attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T,
                       C, NH);
    matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C,
                    3 * C);
    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w,
                       l_ln1_mean, l_ln1_rstd, B, T, C);
  }
  encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B,
                   T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2,
                 float eps, float weight_decay, int t) {
  // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

  // lazily allocate the memory for m_memory and v_memory
  if (model->m_memory == NULL) {
    model->m_memory = (float *)calloc(model->num_parameters, sizeof(float));
    model->v_memory = (float *)calloc(model->num_parameters, sizeof(float));
  }

  for (size_t i = 0; i < model->num_parameters; i++) {
    float param = q115_to_float(model->params_memory[i]);
    float grad = model->grads_memory[i];

    // update the first moment (momentum)
    float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
    // update the second moment (RMSprop)
    float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
    // bias-correct both moments
    float m_hat = m / (1.0f - powf(beta1, t));
    float v_hat = v / (1.0f - powf(beta2, t));

    // update
    model->m_memory[i] = m;
    model->v_memory[i] = v;
    float updated_param =
        param -
        learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    model->params_memory[i] = float_to_q115(updated_param);
  }
}

void gpt2_free(GPT2 *model) {
  free(model->params_memory);
  free(model->grads_memory);
  free(model->m_memory);
  free(model->v_memory);
  free(model->acts_memory);
  free(model->grads_acts_memory);
  free(model->inputs);
  free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

  // build the GPT-2 model from a checkpoint
  // Try to load checkpoints in order of preference:
  // 1. float32 version (gpt2_124M.bin)
  // 2. bfloat16 version (gpt2_124M_bf16.bin)
  // 3. If neither exists, will fall back to random initialization
  GPT2 model;
  const char *checkpoint_files[] = {"gpt2_124M.bin", "gpt2_124M_bf16.bin"};

  int loaded = 0;
  for (int i = 0; i < 2; i++) {
    if (access(checkpoint_files[i], F_OK) != -1) {
      printf("Found checkpoint: %s\n", checkpoint_files[i]);
      gpt2_build_from_checkpoint(&model, checkpoint_files[i]);
      loaded = 1;
      break;
    }
  }

  if (!loaded) {
    printf("No checkpoint files found. Using random initialization.\n");
    gpt2_build_from_checkpoint(&model,
                               "gpt2_124M.bin"); // will trigger fallback
  }

  // build the DataLoaders from tokens files using fineweb dataset
  const char *train_data_pattern = "dev/data/fineweb10B/fineweb_train_*.bin";
  const char *val_data_pattern = "dev/data/fineweb10B/fineweb_val_*.bin";
  int B =
      4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
  int T = 1024; // sequence length 1024 (max for GPT-2)
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
  dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
  printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B * T));
  printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B * T));
  int val_num_batches = 20; // increased for larger dataset

  // build the Tokenizer
  Tokenizer tokenizer;
  tokenizer_init_q15(&tokenizer, "gpt2_tokenizer_q15.bin");

  // some memory for generating samples from the model
  uint64_t rng_state = 1337;
  int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
  const int genT = 64;     // number of steps of inference we will do
  const int steps = 65536; // total training steps

  // train
  struct timespec start, end;
  for (int step = 0; step <= steps; step++) {

    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      printf("val loss %f\n", val_loss);
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
      for (int i = 0; i < B * T; ++i) {
        gen_tokens[i] = tokenizer.eot_token;
      }
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from
        // scratch but the inference here is just for sanity checking anyway and
        // we can maybe optimize a bit more later, with careful tests
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        // furthermore, below we're only using b=0 (i.e. the first row) of all B
        // rows we're in principle running B "inference streams" in parallel
        // here but only using position 0 get the Vp-dimensional vector probs[0,
        // t-1, :]
        float *probs =
            model.acts.probs + (t - 1) * model.config.padded_vocab_size;
        float coin = random_f32(&rng_state);
        // note we're only sampling from the first V elements, ignoring padding
        // (the probabilities in the padded region should be zero anyway)
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        // print the generated token, either using the Tokenizer or a fallback
        if (tokenizer.init_ok) {
          const char *token_str = tokenizer_decode_q15(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          // fall back to printing the token id
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss,
           time_elapsed_s * 1000);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  free(gen_tokens);
  return 0;
}
#endif
