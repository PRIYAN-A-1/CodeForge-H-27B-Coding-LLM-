# Model Compatibility & Hardware Sizing Assessment

**Date:** October 2026  
**Target Hardware Constraints:** NVIDIA GPU (4 GB VRAM ceiling), 16 GB System RAM, Windows host.  
**Long-Term Objective:** 27B-class coding capability.  
**Immediate Objective:** Data-driven fine-tuning to improve the existing coding agent.  

---

## 1. Candidate Model Evaluation Matrix

| Model Identifier | Parameter Count | Precision / Format | Minimum VRAM (Inference) | Minimum VRAM (LoRA Training) | Full Fine-Tuning Feasibility (4GB) | License | Fit Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CodeForge-H Pilot** | **126.94M** | float32 / bfloat16 | ~0.4 GB | **~1.2 GB** | **FEASIBLE** | Apache-2.0 | **Primary In-House Model.** Hybrid [DeltaNet x3 + Attn x1]. Zero external dependencies. |
| **Qwen2.5-Coder-0.5B-Instruct** | **494M** | bfloat16 / int4 | ~0.9 GB | **~2.8 GB** (QLoRA / LoRA) | **FEASIBLE** | Apache-2.0 | **Primary Pretrained Candidate.** High multilingual coding density. Fits within 4GB limit. |
| **Qwen2.5-Coder-1.5B** | 1.54B | bfloat16 / int4 | ~2.2 GB | ~5.8 GB | INFEASIBLE (4GB OOM) | Apache-2.0 | Exceeds 4 GB VRAM ceiling during backward pass even with QLoRA. |
| **Qwen2.5-Coder-7B** | 7.61B | int4 / fp16 | ~4.5 GB | ~16.0 GB | INFEASIBLE | Apache-2.0 | Requires 16 GB+ VRAM for training. |
| **Qwen3.8:27b / 27B-class** | 27B | int4 (GGUF) | ~17 GB RAM | ~72.0 GB VRAM | INFEASIBLE (Local) | Permissive | **Long-Term Target.** Accessible locally for inference via Ollama offload; training requires cloud multi-GPU. |

---

## 2. Hardware Preflight Sizing & Memory Budget

For fine-tuning on a 4.0 GB VRAM GPU:
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{model}} + \text{VRAM}_{\text{gradients}} + \text{VRAM}_{\text{optimizer}} + \text{VRAM}_{\text{activations}} + \text{VRAM}_{\text{workspace}}$$

### Sizing for Qwen2.5-Coder-0.5B with LoRA ($r=8, \alpha=16$):
1. **Model Weights (Frozen 4-bit / 8-bit base):** $494 \times 10^6 \times 0.5 \text{ bytes} \approx 0.25 \text{ GB}$ (or ~1.0 GB in fp16).
2. **LoRA Adapter Parameters (Trainable):** Rank 8 on `q_proj`, `v_proj`: $\approx 4.5 \times 10^6 \text{ params}$.
3. **Optimizer States (AdamW on LoRA only):** $4.5 \times 10^6 \times 8 \text{ bytes} \approx 0.036 \text{ GB}$.
4. **Activations (Batch Size 1, Sequence Length 512 with Gradient Checkpointing):** $\approx 0.85 \text{ GB}$.
5. **Peak Memory Consumption:** $\approx 2.14 \text{ GB}$ (well under the 4.0 GB ceiling).

### Sizing for CodeForge-H Pilot (126.94M):
1. **Model Weights (fp32):** $126.94 \times 10^6 \times 4 \text{ bytes} \approx 0.51 \text{ GB}$.
2. **LoRA / Full Head Gradients:** $\approx 0.25 \text{ GB}$.
3. **Optimizer States:** $\approx 0.51 \text{ GB}$.
4. **Activations (Batch Size 2, Sequence Length 256):** $\approx 0.35 \text{ GB}$.
5. **Peak Memory Consumption:** $\approx 1.62 \text{ GB}$ (operates safely on both 4GB GPU and CPU).

---

## 3. Training Execution Strategy

1. **Immediate Execution (Local):**
   * Train low-rank adapter (`sft_lora_v1.pt`) with assistant-only loss masking on the domain-specific agent dataset.
   * Apply gradient accumulation (steps=4) with per-device batch size=1 to guarantee zero OOM events.
   * Save checkpoints every 10 steps with automatic rollback on loss divergence.

2. **Long-Term Scaling (27B Target):**
   * Keep 27B model serving via Ollama (`qwen3.8:27b`) for general knowledge.
   * The fine-tuned small adapter handles high-frequency agent tasks (Pydantic schemas, Pytest contracts, self-repair loops, and project file generation).
