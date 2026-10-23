# Production Guide: Running the 27B Coding LLM Training Pipeline
**Target Model**: 27-Billion Parameter Coding Language Model  
**Frameworks**: PyTorch Distributed (`torchrun`), DeepSpeed ZeRO-3, Hugging Face Accelerate, BitsAndBytes, PEFT / LoRA  
**Environments Supported**: Local Edge PC (Validation & Smoke Testing) | Multi-GPU Cloud Clusters (Full Scale)

---

## 1. Quick Reference: End-to-End Pipeline Execution

All steps can be executed sequentially or independently:

```bash
# Step 1: Discover candidate datasets & build metadata registry
python scripts/discover_datasets.py

# Step 2: Audit legal licenses, commercial permissions & quarantine test benchmarks
python scripts/validate_licenses.py

# Step 3: Stream and acquire bounded subsets with SHA-256 validation
python scripts/download_datasets.py --resume

# Step 4: Validate encoding, scrub secrets, and detect programming languages
python scripts/validate_data.py

# Step 5: Exact & MinHash LSH deduplication with 13-gram decontamination
python scripts/deduplicate_data.py

# Step 6: Prepare Continued Pretraining & Fill-In-The-Middle (FIM) dataset
python scripts/prepare_pretraining.py

# Step 7: Prepare Supervised Fine-Tuning (SFT) ChatML data with assistant-only loss masking
python scripts/prepare_sft.py

# Step 8: Prepare multi-step Coding-Agent Tool Trajectories
python scripts/prepare_agent_data.py

# Step 9: Run training (Local smoke test or 27B cluster preflight)
python scripts/train.py --smoke-test --steps 10

# Step 10: Run comprehensive benchmark evaluation in sandboxed jail
python scripts/evaluate.py

# Step 11: Export versioned release checkpoint bundle and model card
python scripts/export_model.py
```

---

## 2. Infrastructure Deployment Modes

### Mode A: Local Edge Execution (NVIDIA 4 GB - 8 GB VRAM)
On local workstations with consumer GPUs (e.g., RTX 3050 / 3060 / 4060):
- **Training**: Executed on the CodeForge-H Hybrid Decoder architecture (126.9M parameters) using LoRA ($r=8, \alpha=16$) with micro-batch 1 and gradient accumulation. This validates the entire tokenization, masking, loss reduction, and checkpointing pipeline without CUDA Out-Of-Memory (OOM) errors.
- **Inference**: High-capability 27B inference runs locally via the pre-installed quantized `qwen3.8:27b` (17 GB, Q4_K_M) on the local Ollama server (`http://127.0.0.1:11434`).

```powershell
# Run local 10-step smoke test
python scripts/train.py --model pilot --smoke-test --steps 10

# Resume from checkpoint
python scripts/train.py --model pilot --resume checkpoints/checkpoint_step_10.pt --steps 5
```

---

### Mode B: Cloud / Multi-GPU Server (4x / 8x A100 80GB or H100 80GB)
For training the full 27-billion parameter model weights across multi-GPU nodes:

#### Hardware Requirements:
- **Accelerators**: 8x NVIDIA A100 (80GB SXM4) or 8x NVIDIA H100 (80GB SXM5)
- **Host Memory**: 512 GB System RAM
- **Interconnect**: NVLink (900 GB/s) + InfiniBand NDR (400 Gbps)
- **Storage**: 2 TB NVMe SSD scratch space

#### Execution via Launcher Script:
```bash
# Make script executable
chmod +x scripts/launch_27b_distributed.sh

# Launch 8-GPU distributed training with DeepSpeed ZeRO-3
./scripts/launch_27b_distributed.sh 8 configs/27b_sft_qlora.yaml
```

#### Manual `torchrun` Invocation:
```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train.py \
    --model 27b \
    --config configs/27b_sft_qlora.yaml \
    --deepspeed configs/27b_deepspeed_zero3.json \
    --output-dir checkpoints/27b_release
```

---

## 3. Configuration Reference

### Pretraining (`configs/27b_pretrain.yaml`)
- `vocab_size`: 49152 (Byte-Level BPE with special FIM tokens `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`)
- `d_model`: 5120, `n_layers`: 64, `n_heads`: 40, `n_kv_heads`: 8 (Grouped Query Attention)
- `max_seq_len`: 4096 tokens with RoPE base $100000.0$
- `global_batch_size`: 512 (micro-batch 1 per GPU, gradient accumulation 64 steps)

### Supervised Fine-Tuning QLoRA (`configs/27b_sft_qlora.yaml`)
- `quantization`: NormalFloat4 (NF4) with double quantization
- `compute_dtype`: `bfloat16`
- `lora_r`: 16, `lora_alpha`: 32, `lora_dropout`: 0.05
- `target_modules`: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- `loss_masking`: Assistant-only (`labels = -100` for user/system tokens)

---

## 4. Evaluation & Benchmark Isolation Protocol
To prevent data contamination and ensure benchmark validity:
1. **Quarantined Fixtures**: `HumanEval`, `MBPP`, and `LiveCodeBench` are strictly quarantined from all training splits via `scripts/validate_licenses.py`.
2. **Decontamination Gate**: `scripts/deduplicate_data.py` executes 13-gram overlap filtering against all test prompts.
3. **Sandboxed Jail**: `scripts/evaluate.py` executes all candidate code within disposable, process-bounded sandbox jails with timeout ceilings and dangerous command interception.
