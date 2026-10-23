# CodeForge AI — Hardware Resource & Execution Plan

**Document Version:** 1.0.0  
**Inspection Date:** 2026-10-23  
**Target Architecture:** CodeForge-H (Hybrid Reasoning Decoder LLM & Autonomous Agent)

---

## 1. Executive Summary & Hardware Baseline

Empirical hardware inspection on the host machine confirms:
- **Processor:** AMD Ryzen 5 7235HS (4 physical cores, 8 logical threads)
- **System Memory:** 23.69 GB visible physical RAM (~4.2 GB available at idle)
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, Driver 591.74, CUDA 13.1 hardware support)
- **Python / PyTorch:** Python 3.13.13 with PyTorch 2.14.0+cpu installed
- **Local Inference Engines:** Ollama is running at `http://127.0.0.1:11434` with `qwen3.8:27b` present
- **Storage:** ~66 GB free on drive C:, ~63 GB free on drive G:

---

## 2. Resource Tier Feasibility Analysis

| Tier / Mode | Parameter Target | Memory / Compute Needs | Local Feasibility | Recommended Execution Path |
| :--- | :--- | :--- | :--- | :--- |
| **Mode A: CPU-Only Inference** | 125M – 1.5B (FP16/BF16) or ≤7B (INT4) | 2–6 GB RAM | **Feasible** | Fast and reliable for local testing, mock backend, and small pilot checkpoints. |
| **Mode A: Quantized Local Inference** | 7B – 27B (Q4_K_M / AWQ) | 6–18 GB RAM + VRAM offload | **Constrained** | Ollama with `qwen3.8:27b` (17 GB) runs via hybrid CPU+GPU offload; requires >60s cold-load time and paging when system RAM has <5GB free. Smaller 0.5B–3B local models run in <1.5s. |
| **Mode B: Small-Model GPU Fine-Tuning** | 0.5B – 1.5B (QLoRA 4-bit) | 4–6 GB VRAM | **Feasible with CUDA PyTorch** | 4-bit NF4 QLoRA (rank 16, batch size 1, grad accum 8) fits inside 6 GB VRAM. |
| **Mode B: Larger-Model Adapter Training** | 7B – 27B (LoRA / QLoRA) | 16–36 GB VRAM | **Not Feasible Locally** | 27B QLoRA requires minimum 24 GB VRAM (e.g., RTX 3090/4090 or A10G); 7B QLoRA requires 10–12 GB VRAM. Must use cloud-assisted instances (A100/H100 or RunPod/Lambda). |
| **Mode C: Foundation Model Training (Pilot)** | 125M – 150M | 4–8 GB RAM (CPU) or 4–6 GB VRAM | **Feasible Locally** | Configurable decoder-only Transformer with d_model=768, 12 layers, vocab=49,152 for architectural validation and smoke tests. |
| **Mode C: Foundation Model Pretraining (Full)** | 1B – 27B from scratch | Multiple A100/H100 clusters (10k+ GPU hours) | **Cloud Only** | 27B pretraining requires 100k+ H100 hours. Attempting this on a laptop GPU is strictly prohibited. |

---

## 3. Operational Strategy for CodeForge AI

1. **Development & Testing Baseline:**
   - Use built-in mock inference engine for instant, deterministic CI and unit tests (zero network or model latency).
   - Use local Ollama backend for real natural-language and code-generation agent workflows.
   - Use CPU PyTorch implementation for dataset tokenization, FIM tests, pass@k evaluation fixtures, and small pilot model forward/backward validation.

2. **Fine-Tuning Strategy (Mode B):**
   - Provide standard Hugging Face PEFT + TRL recipes with full verification of LoRA module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).
   - Support CPU-based micro-SFT verification (10 steps) for CI validation and GPU-accelerated QLoRA when CUDA PyTorch is available.

3. **From-Scratch Research Model (Mode C):**
   - Implement the modular decoder-only Transformer architecture (supporting causal attention, RoPE, SwiGLU, RMSNorm, and MTP auxiliary loss).
   - Provide a local pilot configuration (125M params) with reproducible seed (1337) and a dry-run / micro-pretrain loop on curated synthetic code pairs.

4. **Security & Sandbox Isolation:**
   - Because Docker is not installed on the system PATH, CodeForge AI provides a safe process-bounded workspace jail:
     - Disallows commands outside project root.
     - Enforces timeout ceilings (default 10s per command).
     - Caps output buffer at 64 KB.
     - Rejects destructive commands (`rm -rf`, format drives, admin escalation).
