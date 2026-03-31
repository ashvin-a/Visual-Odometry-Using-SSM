---
name: SSM / Mamba research summary
description: Technical notes on State Space Models, Mamba, MambaVO, and MambaGlue relevant to the VO project
type: project
---

## Mathematical foundations
- SSMs: continuous differential equations → discretized to recurrent (inference) or convolutional (training) form
- Three equivalent views: continuous / recurrent / convolution — switching between them is what gives both training parallelism and O(1) inference state
- Discretization methods: Euler (simple), Bilinear/Tustin (better frequency preservation)

## Mamba (S6) key innovations
- **Selective mechanism**: Δ(x_t), B(x_t), C(x_t) are input-dependent (vs. fixed in S4) → model can selectively propagate or forget
- **Hardware-aware scan**: kernel fusion + parallel scan + recomputation → 3× faster than naive on A100
- **Inference**: O(1) hidden state (fixed memory per token, no KV cache)
- **Training**: O(n log n) via parallel scan (comparable to transformer)

## Complexity table
| Architecture | Training | Inference memory |
|---|---|---|
| Transformer | O(n²) | O(n) growing KV cache |
| RNN | O(n) sequential | O(1) but no parallelism |
| SSM/Mamba | O(n log n) | O(1) fixed state |

## SSM limitations
- Non-causal tasks (bidirectional) lose efficiency gains
- Custom CUDA kernels required (NVIDIA-only, not cuDNN)
- Fine-grained token retrieval weaker than transformers
- Training needs FP32 for numerical stability

## Key papers
- Mamba (Gu & Dao, 2023): arXiv:2312.00752
- MambaVO (CVPR 2025): arXiv:2412.20082 — NO code released
- MambaGlue (ICRA 2025): arXiv:2502.00462 — code at github.com/url-kaist/MambaGlue

## MambaGlue architecture
SuperPoint keypoints → MambaAttention Mixer (S6 replaces self-attention, O(N)) → Cross-attention → Deep Confidence Score Regressor with exit mechanism
