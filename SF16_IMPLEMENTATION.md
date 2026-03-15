# SF16 (Super Float 16) Implementation Details

This document outlines the architecture and implementation of **SF16** (Q1.15 Fixed-Point with Block Floating Point scaling) as used in the GPT-2 training pipeline of `superfloat.cuda`.

## 1. Core Format: Q1.15
The primary storage format for tensors is **Q1.15**, a signed 16-bit fixed-point format representing values in the range `[-1, 1)`.

- **Scaling Factor**: $2^{15} = 32768$
- **Resolution**: $\approx 3.05 \times 10^{-5}$
- **Range**: `[-32768, 32767]` (integer) $\rightarrow$ `[-1.0, 0.999969...]` (normalized)

## 2. Block Floating Point (BFP) Scaling
To handle the limited dynamic range of Q1.15, each tensor is associated with a 32-bit floating-point **scale factor**.
$$\text{Actual Value} = \text{Q1.15 Value} \times \text{Scale Factor}$$

### Default Tensor Scales
| Tensor Type | Scale Factor | Rationale |
| :--- | :--- | :--- |
| **Embeddings** | 0.25f | Tokens are typically initialized in a small range. |
| **Attention** | 1.0f | Standard range for Q, K, V and attention outputs. |
| **FFN Activations** | 2.0f | Larger range needed after GELU activations. |
| **Logits** | **24.0f** | **Critical**: Logits require a larger range (approx [-10, 10]) for stable Softmax. |

### Gradient Scales
| Tensor Type | Grad Scale |
| :--- | :--- |
| **Embeddings** | 0.01f |
| **Attention** | 0.1f |
| **FFN** | 0.1f |
| **Logits** | 1.0f |

## 3. Training Stability Optimizations
Scaling alone is insufficient for stable training in fixed-point. Several structural optimizations are implemented to prevent **amplitude collapse** and **loss floor** issues.

### 3.1. RMSNorm-Q (Wait, where's the Mean?)
In SF16 mode, standard **LayerNorm** is replaced with **RMSNorm-Q**.
- **No Mean Subtraction**: $y = \frac{x}{\text{RMS}(x)} \odot \gamma$.
- **Rationale**: Subtracting the mean in Q1.15 precision often destroys small, high-frequency signals that are critical for learning deep representations. RMS-only normalization preserves these signals while providing the necessary variance stabilization.
- **Q-Aware Epsilon**: Uses `1e-3` instead of the standard `1e-5`. Since the smallest non-zero Q1.15 value is $\approx 3 \times 10^{-5}$, a $10^{-5}$ epsilon is effectively invisible and can lead to division by zero or numerical instability.

### 3.2. Selective Weight Decay
Weight decay is only applied to the **FCW** (Feed-Forward Weight) tensors (`tensor_id == 10`).
- **Disabled** on: `wte`, `wpe`, `qkvw`, `attprojw`, `fcprojw`.
- **Reason**: Standard weight decay shrinks weights toward zero. In Q1.15, this leads to loss of precision and entropy increase in attention logits, causing a training loss floor.

### 3.3. Residual Branch Scaling
To prevent amplitude collapse in deep networks, residual branches are scaled before addition:
- **Attention Residual**: 0.65f
- **MLP Residual**: 0.75f
$$\text{Output} = \text{Residual} + (\text{Branch} \times \text{Scale})$$

### 3.4. Layer-Wise Gradient Scaling
Gradients are scaled inversely with layer depth to combat vanishing gradients:
- **Range**: [1.0, 1.5]
- **Earlier Layers** (closer to input) receive larger gradient scales.

### 3.5. Positional Embedding Freezing
Position embeddings (`wpe`) are **frozen** after a set number of steps (`Q115_WPE_FREEZE_STEP = 3000`).
- **Reason**: Stabilizes the base coordinate system of the model, allowing representational budget to focus on semantic content.

### 3.6. Numerical Safety & Clamping
- **Activation Clamping**: All activations are clamped to `[-3.0, 3.0]` after Norm layers. This prevents "spikes" from saturating the fixed-point range.
- **Weight Clamping**: Weights are clamped to the Q1.15 range `[-1, 1)` after every AdamW update to ensure they remain representable without overflow.

## 4. Arithmetic & Optimization Implementation
- **Multiplication**: `(int32_t(a) * int32_t(b)) >> 15` with saturation.
- **Accumulation**: Performed in **FP32** for high-precision dot products before quantizing back to Q1.15 for storage. This ensures that the bulk of the compute (MatMuls) maintains high internal precision.
- **AdamW Updates**: Performed on **FP32 master weights** using standard AdamW logic, then deterministic quantization back to Q1.15 for the next forward pass.

## 5. Summary Comparison: SF16 vs Standard FP32
| Feature | Standard FP32 | SF16 (Q1.15) |
| :--- | :--- | :--- |
| **Weight Storage** | 32-bit | 16-bit (Fixed) |
| **Activation Storage** | 32-bit (or BF16) | 16-bit (Fixed) |
| **Normalization** | LayerNorm | RMSNorm-Q |
| **Residual Path** | $x + \text{MLP}(x)$ | $x + 0.75 \times \text{MLP}(x)$ |
| **Weight Decay** | All 2D Weights | **Only** FCW (`fc_proj`) |
| **Epsilon** | $10^{-5}$ | $10^{-3}$ |

## 6. How to Run SF16
To compile and train with SF16 (Q1.15) enabled:

```bash
# Compile with Q115 enabled (default in mainline superfloat)
make train_gpt2cu

# Run training
./train_gpt2cu -b 16 -t 1024 -x 10000
```

The training progress will show logs indicating Q1.15 initialization and scaling:
`Initializing model with random Q1.15 weights`
