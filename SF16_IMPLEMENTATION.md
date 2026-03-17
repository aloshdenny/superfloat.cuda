# SF16 (Q115) Implementation Details

This document describes the current SF16/Q115 implementation in this repository.
It focuses on what is actually compiled and executed in the current codebase,
especially for the true-forward path enabled by `SF16_TRUE_FORWARD=1`.

## 1. Scope and Terminology

- SF16 in this repo means Q1.15 boundary simulation (`simulate_q115`) applied to
	tensors while running on NVIDIA-friendly native kernels.
- Q115 mode is enabled with `-DENABLE_Q115`.
- True-forward behavior is enabled with `-DSF16_TRUE_FORWARD=1`.

Important: this is not a pure int16 hardware matmul pipeline. In Q115 mode,
`floatX` is mapped to BF16 (`__nv_bfloat16`) and SF16 boundaries are enforced by
explicit quantize/dequantize steps at key write boundaries.

## 2. Core Numeric Format (Q1.15)

Q1.15 represents normalized values in `[-1, 1)` with signed 16-bit fixed-point
semantics.

- Scale factor: $2^{15} = 32768$
- Resolution: $1 / 32768 \approx 3.05 \times 10^{-5}$
- Integer range: `[-32768, 32767]`
- Float range: `[-1.0, 0.999969...]`

In code:

- Overflow threshold is `Q115_OVERFLOW_THRESHOLD = 0.999f`.
- `float_to_q115` clamps to `[-Q115_OVERFLOW_THRESHOLD, Q115_OVERFLOW_THRESHOLD]`.
- `simulate_q115(x)` performs quantize+dequantize roundtrip:
	`q115_to_float(float_to_q115(x))`.

## 3. Storage and Compute Model

Under `ENABLE_Q115`:

- Tensor storage type (`floatX`) is BF16 (`__nv_bfloat16`).
- cuBLAS low-precision datatype (`CUBLAS_LOWP`) is BF16.
- For `ENABLE_Q115 && SF16_TRUE_FORWARD`, compute mode is
	`CUBLAS_COMPUTE_32F_FAST_16BF` (with compatibility fallback macro).

This design keeps RTX 4090 performance high while enforcing SF16 semantics at
explicit forward boundaries.

## 4. Forward Boundary Enforcement

### 4.1 Matmul Outputs

Forward matmul outputs are boundary-quantized by `q115_simulate_kernel`:

- In `SF16_TRUE_FORWARD`, the kernel does:
	1. `simulate_q131(v)` (SF32 register simulation)
	2. `simulate_q115(v)` (SF16 storage boundary)
- The kernel is applied for forward outputs (`!backward`) after cuBLASLt matmul.

### 4.2 Attention

For `ENABLE_Q115`:

- Attention dynamic expansion scale is neutralized: `att_scale = 1.0f`.
- Softmax computation reads through `simulate_q115(...)`.
- Softmax writeback is quantized to SF16 boundary:
	`__stcs(..., (floatX)simulate_q115(ev * norm));`

The standard attention temperature factor $1/\sqrt{HS}$ remains part of normal
attention math.

### 4.3 Classifier / Logits

In `fused_classifier.cuh`:

- If `SF16_TRUE_FORWARD` is defined, `Q115_LOGIT_SCALE = 1.0f`.
- Otherwise it falls back to legacy `Q115_LOGITS_SCALE` behavior.
- Logit reads are through `simulate_q115(...)`.

### 4.4 Norm and Residual Paths

Norm/residual paths clamp and quantize in Q115 mode:

- Activation clamps use:
	- `Q115_ACTIVATION_CLAMP_MIN = -Q115_OVERFLOW_THRESHOLD`
	- `Q115_ACTIVATION_CLAMP_MAX = +Q115_OVERFLOW_THRESHOLD`
- Outputs are written with `simulate_q115(...)` in the key norm/residual forward
	kernels.

Residual branch scales are mode-dependent:

- `SF16_TRUE_FORWARD`: `Q115_ATTENTION_RESIDUAL_SCALE = 1.0f`,
	`Q115_MLP_RESIDUAL_SCALE = 1.0f`
- Legacy mode: `0.65f` and `0.75f`

### 4.5 GELU and Encoder

- GELU forward reads via `simulate_q115` and writes quantized outputs via
	`simulate_q115`.
- Encoder forward sum (`wte + wpe`) is quantized with `simulate_q115` before
	store.

## 5. Legacy BFP Scale Constants (Still Present)

`q115_common.cuh` still defines legacy BFP-oriented constants such as:

- `Q115_EMBEDDING_SCALE`
- `Q115_ATTENTION_SCALE`
- `Q115_FFN_SCALE`
- `Q115_LOGITS_SCALE` (24.0f)

These constants remain available for legacy/non-true-forward experiments and
helper paths, but true-forward Q115 explicitly neutralizes the major forward
expansion scales where required.

## 6. cuDNN Interaction

To preserve strict Q115/SF16 boundary semantics in attention, the code now
rejects Q115+cuDNN builds at compile time:

```c
#if defined(ENABLE_Q115) && defined(ENABLE_CUDNN)
#error "ENABLE_CUDNN is not supported with ENABLE_Q115. Disable USE_CUDNN for Q115/SF16 builds."
#endif
```

Also, Make defaults to:

- `USE_CUDNN ?= 0`

## 7. Current Build and Run Commands

Use Q115-specific targets (not `train_gpt2cu`) for SF16/Q115 mode:

```bash
# GPT-2: Q115 true-forward
make train_gpt2q115cu

# GPT-2: Q115 true-forward + weight-constrained mode
make train_gpt2q115_constrainedcu

# GPT-3 (standalone): pure Q115 true-forward
make train_gpt3q115cu

# Optional dry-run to inspect compile flags
make -n train_gpt2q115cu | head -n 40
```

Expected key compile defines for Q115 true-forward:

- `-DENABLE_Q115`
- `-DSF16_TRUE_FORWARD=1`

Run example:

```bash
./train_gpt2q115cu -b 16 -t 1024 -x 10000

# Standalone GPT-3 pure SF16
./train_gpt3q115cu -e "gpt3:c768" -Q 115 -b 16 -t 2048 -x 10000
```

For the full GPT-3 125M 300B-token run, use:

- `scripts/run_gpt3_125M.sh`

That script now builds and runs `train_gpt3q115cu` by default,
keeps cuDNN off for Q115, and passes `-Q 115` to verify runtime precision.

## 8. Quick Reality Check

If you need to verify strict forward semantics quickly:

1. Confirm Q115 target includes `-DENABLE_Q115 -DSF16_TRUE_FORWARD=1`.
2. Confirm attention softmax store uses `simulate_q115(ev * norm)`.
3. Confirm matmul post-kernel applies `simulate_q131` then `simulate_q115`.
4. Confirm cuDNN is disabled for Q115 (or compile-time guard is present).

This reflects the current implementation state in this branch.
