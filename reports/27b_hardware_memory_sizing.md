# Hardware Memory Sizing & SFT/Pretraining Feasibility for 27B Coding Model
**Date**: 2026-10-23  
**Target Model**: 27-Billion Parameter Coding LLM (e.g., CodeForge-H XL 27B / Qwen3.8-27B / Qwen2.5-Coder-27B)  
**Host Environment**: Windows 11 Host, NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM), 16 GB System RAM

---

## 1. Executive Summary & Mathematical Proof of Feasibility
A 27-billion parameter language model represents a powerful foundation tier for autonomous software engineering agents. However, executing backward passes and updating optimizer states requires substantial accelerator memory.

As mandated by Section 8 & 13 of the Master Specification:
> *"For a 27B model, calculate actual memory requirements before selecting a configuration. Include model weights, optimizer states, gradients, activations, temporary buffers, and checkpoint storage. Do not assume that ordinary single-GPU full fine-tuning is feasible. If the environment is insufficient, automatically switch to a feasible LoRA/QLoRA or smaller-subset experiment when compatible, or produce a documented multi-GPU/cloud execution plan. Never silently pretend that training has completed."*

### Mathematical Breakdown by Precision and Training Mode:

| Configuration | Precision | Weights | Optimizer States (AdamW) | Gradients | Activations ($L=4096, B=1$) | Total VRAM Required | Feasible on 4 GB RTX 3050? | Target Infrastructure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning (FP32)** | FP32 | 108.0 GB | 216.0 GB | 108.0 GB | 48.0 GB | **480.0 GB** | **No** (Exceeds by 120x) | $8 \times \text{H100 } 80\text{GB}$ |
| **Full Fine-Tuning (BF16)** | BF16 | 54.0 GB | 108.0 GB | 54.0 GB | 28.0 GB | **244.0 GB** | **No** (Exceeds by 61x) | $4 \times \text{A100 } 80\text{GB}$ (ZeRO-3) |
| **LoRA SFT (BF16, $r=16$)** | BF16 Base | 54.0 GB | 0.22 GB | 0.11 GB | 12.0 GB | **66.3 GB** | **No** (Exceeds by 16.5x) | $1 \times \text{A100 } 80\text{GB}$ or $2 \times \text{RTX 4090 } 24\text{GB}$ |
| **QLoRA SFT (INT4, $r=16$)** | NF4 Base | 14.85 GB | 0.22 GB | 0.11 GB | 4.5 GB | **19.7 GB** | **No** (Exceeds by 4.9x) | $1 \times \text{RTX 3090/4090 } 24\text{GB}$ or A10G 24GB |
| **ZeRO-3 + CPU Offload** | BF16 | 16.0 GB VRAM | Offloaded to RAM | Offloaded | 8.0 GB | **16 GB VRAM + 128 GB RAM** | **No** (Host RAM is 16 GB, not 128 GB) | Workstation with 128GB RAM |

---

## 2. Autonomous Dual-Track Solution

To provide the user with a 100% legal, reproducible, and executable system without fabricating training results:

### Track A: Local Runnable Pipeline & Smoke Testing (Edge Feasible)
- **Local Validation Model**: The modular CodeForge-H Hybrid Decoder architecture (126.9M parameters) is used to validate all data ingestion, tokenizer alignment, assistant-only loss masking, Fill-In-The-Middle (FIM) pretraining, LoRA adaptation, and checkpoint export on the local 4 GB VRAM GPU without triggering Out-Of-Memory (OOM) errors.
- **Local 27B Inference**: The pre-installed quantized `qwen3.8:27b` (17 GB, Q4_K_M) on the local Ollama server (`http://127.0.0.1:11434`) is utilized for high-reasoning coding-agent generation, code review, and multi-file architecture synthesis.

### Track B: Turnkey Cloud / Multi-GPU 27B Training Launch Suite
- Fully configured DeepSpeed ZeRO-3 configuration: [`configs/27b_deepspeed_zero3.json`](file:///g:/My%20Drive/thoooo/configs/27b_deepspeed_zero3.json).
- Distributed QLoRA and FSDP training configs: [`configs/27b_sft_qlora.yaml`](file:///g:/My%20Drive/thoooo/configs/27b_sft_qlora.yaml).
- Production multi-GPU launcher script: [`scripts/launch_27b_distributed.sh`](file:///g:/My%20Drive/thoooo/scripts/launch_27b_distributed.sh).
- Supports 4x/8x A100 or H100 GPU clusters with PyTorch Distributed (torchrun / Accelerate).
