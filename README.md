[README.md](https://github.com/user-attachments/files/26256900/README.md)
# ML/DL Architecture Building Blocks Library

A comprehensive catalog and reference implementation of every known deep learning trick and architecture component, organized for modular model building on MLX / Apple Silicon.

---

## Catalog

### Attention Mechanisms (`attention/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Multi-Head Attention | `multi_head.py` | Attention Is All You Need (Vaswani et al.) | 2017 | Standard scaled dot-product attention with multiple heads. Foundation of all transformer variants. | Yes (base) |
| Grouped-Query Attention | `grouped_query.py` | GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al.) | 2023 | KV heads shared across query head groups; interpolates between MHA and MQA. Reduces KV cache by group factor. | **Yes** |
| Multi-Query Attention | `multi_query.py` | Fast Transformer Decoding (Shazeer) | 2019 | Single KV head shared across all query heads. Maximum KV cache savings. | No |
| Sliding Window Attention | `sliding_window.py` | Longformer (Beltagy et al.) / Mistral | 2020/2023 | Each token attends only to a local window. O(N*W) instead of O(N^2). Enables long contexts cheaply. | **Yes** |
| Flash Attention | `flash.py` | FlashAttention (Dao et al.) | 2022 | IO-aware exact attention; tiles computation to stay in SRAM. No approximation, 2-4x speedup. | Concept only |
| Linear Attention | `linear.py` | Transformers are RNNs (Katharopoulos et al.) | 2020 | Replaces softmax with kernel feature maps for O(N) attention via associativity trick. | No |
| Sparse Attention | `sparse.py` | BigBird (Zaheer et al.) / Longformer | 2020 | Combines local window + global tokens + random attention. O(N) with long-range coverage. | No |
| ALiBi Attention Bias | `alibi.py` | Train Short Test Long (Press et al.) | 2022 | Linear distance bias instead of positional embeddings. Extrapolates to longer sequences than seen in training. | No |
| Mamba / SSM Hybrid | `mamba.py` | Mamba (Gu & Dao) | 2023 | Selective state-space model; input-dependent dynamics replace attention. O(N) with hardware-aware scan. | No |

### Normalization (`normalization/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Layer Normalization | `layer_norm.py` | Layer Normalization (Ba et al.) | 2016 | Normalize across features per-sample. Standard in transformers. | No (use RMS) |
| RMS Normalization | `rms_norm.py` | Root Mean Square Layer Normalization (Zhang & Sennrich) | 2019 | Simplified LayerNorm without mean centering. Faster, works as well. | **Yes** |
| Group Normalization | `group_norm.py` | Group Normalization (Wu & He) | 2018 | Normalize across channel groups. Batch-size independent. Mainly for vision. | No |
| DeepNorm | `deep_norm.py` | DeepNet (Wang et al.) | 2022 | Residual scaling + special init for stable training of very deep transformers (1000+ layers). | No |

### Activation Functions (`activations/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| SwiGLU | `swiglu.py` | GLU Variants Improve Transformer (Shazeer) | 2020 | Swish-gated linear unit. Best-performing GLU variant for FFN. | **Yes** |
| GEGLU | `geglu.py` | GLU Variants Improve Transformer (Shazeer) | 2020 | GELU-gated linear unit. Close to SwiGLU performance. | No |
| ReLU Squared | `relu_squared.py` | Primer (So et al.) | 2021 | ReLU(x)^2. Simple, found by architecture search. Surprisingly effective. | No |
| Mish | `mish.py` | Mish: A Self Regularized Non-Monotonic Activation (Misra) | 2019 | x * tanh(softplus(x)). Smooth, non-monotonic. Popular in vision. | No |
| GELU | `gelu.py` | Gaussian Error Linear Units (Hendrycks & Gimpel) | 2016 | Gaussian-weighted ReLU. Default in BERT/GPT-2. | No (use SwiGLU) |

### Positional Embeddings (`embeddings/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Rotary Position Embedding (RoPE) | `rope.py` | RoFormer (Su et al.) | 2021 | Encodes position via rotation matrices in complex space. Relative, decays with distance. | **Yes** |
| ALiBi | `alibi.py` | Train Short Test Long (Press et al.) | 2022 | No positional embedding; adds linear bias to attention scores. Zero extra params. | No |
| Learned Positional Embedding | `learned.py` | Attention Is All You Need (Vaswani et al.) | 2017 | Trainable embedding table indexed by position. Simple, limited to trained length. | No |
| Sinusoidal Positional Embedding | `sinusoidal.py` | Attention Is All You Need (Vaswani et al.) | 2017 | Fixed sin/cos embeddings. Theoretically extrapolates but practically limited. | No |
| Value Embeddings | `value_embeddings.py` | nGPT / custom | 2024 | Per-layer learnable embeddings added to values. Gives each layer a unique "perspective". | **Yes** |

### Optimizers (`optimizers/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Muon | `muon.py` | Muon (Jordan et al.) | 2024 | Momentum + orthogonalization via Newton-Schulz iteration. Excellent for pre-training. | **Yes** |
| Lion | `lion.py` | Symbolic Discovery of Optimization Algorithms (Chen et al.) | 2023 | Sign-based optimizer found by program search. Lower memory than Adam. | No |
| Sophia | `sophia.py` | Sophia (Liu et al.) | 2023 | Second-order optimizer using diagonal Hessian estimate. Clips per-coordinate. | No |
| Schedule-Free | `schedule_free.py` | The Road Less Scheduled (Defazio & Mishchenko) | 2024 | No learning rate schedule needed. Averages iterate and evaluation point. | No |
| Shampoo | `shampoo.py` | Scalable Second Order Optimization (Gupta et al.) | 2018 | Full-matrix preconditioning via Kronecker factorization. State-of-art convergence. | No |

### Training Utilities (`training/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Gradient Checkpointing | `gradient_checkpointing.py` | Training Deep Nets with Sublinear Memory Cost (Chen et al.) | 2016 | Trade compute for memory by recomputing activations during backward pass. | Partial |
| Mixed Precision | `mixed_precision.py` | Mixed Precision Training (Micikevicius et al.) | 2018 | FP16/BF16 forward + FP32 master weights. 2x memory savings + speed. | Partial |
| Curriculum Learning | `curriculum_learning.py` | Curriculum Learning (Bengio et al.) | 2009 | Train on easy examples first, gradually increase difficulty. | No |
| Data Mixing | `data_mixing.py` | DoReMi (Xie et al.) | 2023 | Dynamic domain weights during pre-training for optimal data mixture. | No |
| Warmup Strategies | `warmup_strategies.py` | Various | 2017+ | LR warmup patterns: linear, cosine, WSD, etc. Critical for stable training. | **Yes** (WSD) |
| Loss Functions | `loss_functions.py` | Various | - | Cross-entropy, focal loss, label smoothing, z-loss, etc. | **Yes** (CE) |

### Architecture Blueprints (`architectures/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Transformer | `transformer.py` | Attention Is All You Need (Vaswani et al.) | 2017 | Standard decoder-only transformer. The baseline. | **Yes** |
| Mamba (SSM) | `mamba.py` | Mamba (Gu & Dao) | 2023 | Selective state-space model. O(N) sequence, no attention. Strong on language. | No |
| RWKV | `rwkv.py` | RWKV (Peng et al.) | 2023 | Linear-attention RNN. Trains like transformer, infers like RNN. O(N) total. | No |
| Hyena | `hyena.py` | Hyena Hierarchy (Poli et al.) | 2023 | Long convolutions + gating. Sub-quadratic, no attention. | No |
| RetNet | `retnet.py` | Retentive Network (Sun et al.) | 2023 | Retention mechanism: parallel training, recurrent inference, chunk-wise hybrid. | No |
| Mixture of Experts | `mixture_of_experts.py` | Switch Transformers (Fedus et al.) | 2022 | Sparse MoE: route tokens to top-K experts. Scale params without scaling compute. | No |

### Tricks (`tricks/`)

| Technique | File | Paper | Year | Description | In Our Model? |
|-----------|------|-------|------|-------------|---------------|
| Weight Tying | `weight_tying.py` | Using the Output Embedding (Press & Wolf) | 2017 | Share input embedding and output projection weights. Saves params + improves quality. | No |
| Logit Soft-Capping | `logit_softcap.py` | Gemma 2 (Google) | 2024 | Tanh-based capping of logits/attention scores to prevent divergence. | **Yes** |
| Z-Loss | `z_loss.py` | PaLM (Chowdhery et al.) | 2022 | Penalize large logits to stabilize softmax. Prevents log-sum-exp overflow. | No |
| Auxiliary Loss | `auxiliary_loss.py` | Switch Transformers (Fedus et al.) | 2022 | Load-balancing loss for MoE routing. Prevents expert collapse. | No |
| Spectral Normalization | `spectral_norm.py` | Spectral Normalization (Miyato et al.) | 2018 | Constrain weight matrix spectral norm to 1. Stabilizes training, especially GANs. | No |
| Stochastic Depth | `stochastic_depth.py` | Deep Networks with Stochastic Depth (Huang et al.) | 2016 | Randomly drop entire layers during training. Regularization + faster training. | No |
| QK Normalization | `qk_norm.py` | Scaling Vision Transformers (Dehghani et al.) | 2023 | Normalize Q and K before dot product. Prevents attention logit growth in deep nets. | **Yes** |

---

## Quick Usage

```python
import mlx.core as mx
from lib.ml_blocks.attention.grouped_query import GroupedQueryAttention
from lib.ml_blocks.activations.swiglu import SwiGLU
from lib.ml_blocks.normalization.rms_norm import RMSNorm

# Build a transformer block
norm = RMSNorm(dims=1024)
attn = GroupedQueryAttention(dims=1024, n_heads=16, n_kv_heads=4)
ffn = SwiGLU(dims=1024, hidden_dims=2816)

x = mx.random.normal((1, 128, 1024))
x = x + attn(norm(x))
x = x + ffn(norm(x))
```

## Design Principles

1. **MLX-native**: All implementations use `mlx.core` and `mlx.nn` primitives
2. **Minimal dependencies**: Only MLX + standard library
3. **Breadth over depth**: Know WHAT exists; implementations are reference-quality skeletons
4. **Paper-linked**: Every technique links to its source paper for deep dives
5. **Drop-in ready**: Components follow consistent interfaces for easy model assembly
