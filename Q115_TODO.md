# Q1.15 Implementation TODO List

## ✅ Completed Optimizations (Breaking the ~4.4 Loss Floor)

The following optimizations have been implemented to push Q1.15 training toward the theoretical floor of ~3.4 loss:

### 1. ✅ Increased Logit Scale (CRITICAL) - Expected gain: 0.3-0.6 loss
- **File**: `llmc/q115_common.cuh`
- **Change**: `Q115_LOGITS_SCALE` increased from 16.0 to **24.0**
- **Why**: Logits were too small due to bounded weights → softmax entropy too high → loss floor
- **Effect**: Equivalent to lowering softmax temperature, restores expressivity

### 2. ✅ Q-aware LayerNorm Epsilon - Expected gain: 0.15-0.25 loss
- **File**: `llmc/q115_common.cuh`, `llmc/layernorm.cuh`
- **Change**: `Q115_LAYERNORM_EPS` set to **1e-3** (default 1e-5 is below Q1.15 resolution ~3e-5)
- **Why**: Prevents rstd explosion, avoids denorm flushing, improves gradient signal alignment

### 3. ✅ Position Embedding Freezing - Expected gain: 0.1-0.2 loss
- **File**: `train_gpt2.cu`
- **Change**: After **3000 steps** (`Q115_WPE_FREEZE_STEP`), wpe updates are frozen
- **Why**: Frees dynamic range for semantic embeddings, reduces interference noise

### 4. ✅ RMSNorm Instead of LayerNorm (Already implemented) - Expected gain: 0.3-0.5 loss
- **File**: `llmc/layernorm.cuh`
- **Change**: When `ENABLE_Q115` is defined, uses RMSNorm (no mean subtraction)
- **Why**: No mean subtraction avoids destroying small signals in fixed-point

### 5. ✅ Explicit Residual Branch Scaling - Expected gain: 0.15-0.3 loss
- **Files**: `llmc/layernorm.cuh`, `llmc/q115_common.cuh`
- **Change**: Residual adds now use scaled branches:
  - Attention residual: `Q115_ATTENTION_RESIDUAL_SCALE = 0.65`
  - MLP residual: `Q115_MLP_RESIDUAL_SCALE = 0.75`
- **Why**: Prevents amplitude collapse, preserves head-wise variance

### 6. ✅ Disabled Weight Decay on QKV Matrices - Expected gain: 0.1-0.2 loss
- **File**: `train_gpt2.cu`
- **Change**: Weight decay now ONLY applies to `fcw` (tensor 10)
- **Disabled on**: wte(0), wpe(1), qkvw(4), attprojw(6), fcprojw(12)
- **Why**: Decay shrinks attention logits → entropy ↑, Q1.15 has no headroom for shrinkage

### 7. ✅ Embedding Initialization Scaling - Expected gain: 0.1 loss
- **File**: `train_gpt2.cu`, `llmc/q115_common.cuh`
- **Change**: At init:
  - wte scaled by `Q115_WTE_INIT_SCALE = 1.2` (scale up)
  - wpe scaled by `Q115_WPE_INIT_SCALE = 0.8` (scale down)
  - qkvw scaled by `Q115_QKV_INIT_SCALE = 1.3` (increase head diversity)
- **Why**: Rebalances representational budget, improves early token separation

### 8. ✅ Activation Clamping After LN - Expected gain: 0.1-0.15 loss
- **Files**: `llmc/layernorm.cuh`, `llmc/q115_common.cuh`
- **Change**: Activations clamped to `[-3.0, 3.0]` after normalization
- **Why**: Prevents rare spikes from poisoning fixed-point math, improves gradient stationarity

---

## Expected Final Loss Targets

| Optimizations Applied | Expected Loss |
|----------------------|---------------|
| Items 1-6 only | **3.6 - 3.9** |
| All items 1-8 | **~3.4** (theoretical Q1.15 floor) |

**Note**: Loss below ~3.3 is NOT achievable without:
- Weight precision increase
- Architectural changes (MoE, μParam, rotary embeddings, etc.)

---

## Remaining Tasks (Lower Priority)

### 1. Fix Gradient Activation Structure
**Problem**: `grads_acts` needs to be all-float but points to mixed-type `ActivationTensors`

**Solution**: Create a separate `ActivationTensorsGrad` struct (all float):
```c
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    // ... all other activations as float
} ActivationTensorsGrad;
```

Then update `GPT2` struct:
```c
ActivationTensorsGrad grads_acts;
```

### 2. Create Float Activation Allocator
```c
float* malloc_and_point_activation_grads(ActivationTensorsGrad* acts, size_t* act_sizes) {
    // Similar to current allocator but all float
}
```

### 3. Update gpt2_zero_grad()
```c
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { 
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); 
    }
    if(model->grads_acts_memory != NULL) { 
        // Calculate total float activation size
        size_t total_float_acts = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            total_float_acts += model->act_sizes[i];
        }
        memset(model->grads_acts_memory, 0, total_float_acts * sizeof(float)); 
    }
}
```

### 4. Update gpt2_update() Optimizer
**Problem**: Need to update Q1.15 parameters from float gradients

```c
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // ... Adam computation in float ...
    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = q115_to_float(model->params_memory[i]);
        float grad = model->grads_memory[i];
        
        // Update m and v (float)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        
        // Bias correction
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));
        
        // Update in float
        float updated_param = param - learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        
        // Store updated moments
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        
        // Convert back to Q1.15
        model->params_memory[i] = float_to_q115(updated_param);
    }
}
```

### 5. Update gpt2_backward() Function Calls
All the backward function calls in `gpt2_backward()` need to be checked for proper type usage:

- `crossentropy_softmax_backward()`: ✅ Already correct (dlogits: float, logits: Q1.15)
- `matmul_backward()`: ✅ Already updated
- `layernorm_backward()`: ✅ Already updated
- `residual_backward()`: ✅ Already updated
- `attention_backward()`: ✅ Already updated
- `gelu_backward()`: ✅ Already updated
- `encoder_backward()`: ✅ Already correct

### 6. Update gpt2_forward() Function Calls
All forward function calls need to be verified:

```c
// Layer loop - get Q1.15 pointers correctly
float* l_ln1w = params.ln1w + l * C;  // Should be: q115_t* l_ln1w = ...
```

All pointer arithmetic in gpt2_forward() for params should use `q115_t*` instead of `float*`.

### 7. Fix crossentropy_softmax_backward() Signature
Currently expects float logits, but we have Q1.15:

```c
void crossentropy_softmax_backward(float* dlogits,  // output gradients (float)
                                    float* dlosses, 
                                    float* probs,    // input probs (float)
                                    int* targets,
                                    int B, int T, int V, int Vp);
```

This is actually correct - `dlogits` are output gradients (float), not related to forward Q1.15 logits.

### 8. Compile and Test
```bash
cd /Users/aloshdenny/vscode/llm.c
make train_gpt2
```

Expected errors to fix:
- Type mismatches in gpt2_forward() pointer arithmetic
- Missing ActivationTensorsGrad definition
- Memory allocation size calculations

## Quick Reference: Type Summary

### Parameters:
- Storage: `q115_t*` (Q1.15)
- Gradients: `float*`
- Optimizer: `float*` (m_memory, v_memory)

### Activations (forward):
- Mixed: Some `q115_t*`, some `float*` (see ActivationTensors)

### Activation Gradients (backward):
- All: `float*`

### Intermediate Computations:
- Matmul accumulation: `float`
- Attention scores: `float`
- LayerNorm stats: `float`
- Softmax: `float`

## Testing Checklist

- [ ] Code compiles without errors
- [ ] Code compiles without warnings
- [ ] Random initialization runs
- [ ] Checkpoint loading works (converts to Q1.15)
- [ ] Forward pass completes
- [ ] Backward pass completes
- [ ] Optimizer update runs
- [ ] Loss decreases over iterations
- [ ] Generated text is coherent
- [ ] No NaN/Inf values in training
- [ ] Memory usage is ~50% of original

## Notes

1. The conversion factor (0.5x for checkpoints, 0.1 range for random init) may need tuning
2. Monitor training curves carefully for signs of precision loss
3. Consider adding saturation counters for debugging
4. May want to add runtime flag to disable Q1.15 (compile-time)
