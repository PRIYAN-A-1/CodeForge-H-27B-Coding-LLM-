# Setup, Training, and Recovery Guide: Autonomous Model Integration
**Project**: CodeForge AI  
**Target Architecture**: Multilingual Coding Agent with CodeForge-H LoRA Fine-Tuning  
**Hardware Profile**: NVIDIA GPU (4 GB VRAM ceiling), 16 GB System RAM, Windows Host ($0 Cloud Spend)

---

## 1. System Overview & Architecture
This guide provides complete instructions for executing the autonomous data collection, preprocessing, LoRA fine-tuning, evaluation, integration, and continuous improvement workflow for CodeForge AI.

The system is architected to operate within a conservative **4 GB VRAM** envelope while maintaining full multi-file software engineering, sandboxed execution, and automated code repair capabilities.

```
                    +--------------------------------+
                    |  External Sources (Approved)   |
                    |  (Magicoder, CodeFeedback, etc)|
                    +----------------+---------------+
                                     |
                                     v
                        [ ml/data/collection ]
                                     |
                                     v
                       datasets/raw/*.jsonl (Raw)
                                     |
                                     v
                       [ ml/data/preprocessing ]
                   (Deduplication + Decontamination)
                                     |
                                     v
                      datasets/processed/train.jsonl
                                     |
                                     v
                        [ ml/training/finetuning ]
                        (LoRA r=8, alpha=16 SFT)
                                     |
                                     v
                  ml/training/checkpoints/*.pt (Adapter)
                                     |
                                     v
                        [ ml/evaluation/harness ]
                       (Validation Loss & Sandbox)
                                     |
                                     v
                    [ ml/inference & Model Registry ]
                      (Dynamic Routing + Rollback)
                                     |
                                     v
                        [ Continuous Feedback Loop ]
```

---

## 2. Environment Verification & Preflight
Before running any data acquisition or training operations, execute the hardware preflight check:

```bash
python -c "from ml.training.finetuning.train_sft import TrainingPreflight; import json; print(json.dumps(TrainingPreflight.verify(), indent=2))"
```

Expected output:
- `hardware_approved`: `true`
- `vram_safe_ceiling_gb`: `4.0`
- `device`: `"cuda"` (or CPU fallback if CUDA unavailable)
- `tokenizer_ready`: `true`

---

## 3. Autonomous Dataset Pipeline

### Step 3.1: Download Approved Datasets
Run the automated downloader to fetch bounded, permissively-licensed subsets:
```bash
python ml/data/collection/download_datasets.py
```
- Fetches bounded samples from approved permissive sources (MIT/Apache-2.0).
- Calculates SHA-256 integrity checksums.
- Writes raw data to `datasets/raw/`.
- Updates `reports/dataset_manifest.json`.

### Step 3.2: Clean, Deduplicate & Decontaminate
Run the preprocessing engine:
```bash
python ml/data/preprocessing/prepare_dataset.py
```
- Performs exact SHA-256 deduplication and MinHash LSH near-deduplication.
- Filters PII, AWS/GitHub secrets, and malformed files.
- Executes 13-gram decontamination against HumanEval and MBPP.
- Produces balanced splits:
  - `datasets/processed/train.jsonl`
  - `datasets/processed/validation.jsonl`
  - `datasets/evaluation/eval_heldout.jsonl`
- Outputs `reports/dataset_quality_report.json` and `reports/token_count_report.json`.

---

## 4. Model Training (SFT with LoRA)

Execute the supervised fine-tuning pipeline:
```bash
python ml/training/finetuning/train_sft.py
```
- **Configuration**:
  - Base Model: CodeForge-H Hybrid Decoder (126.9M parameters)
  - LoRA Targets: `q_proj`, `v_proj` ($r=8, \alpha=16$)
  - Micro-Batch Size: 1, Gradient Accumulation: 4 steps
  - Assistant-Only Loss Masking: User/System tokens masked with `-100`
- **Output Artifact**:
  - Adapter Checkpoint: `ml/training/checkpoints/codeforge_lora_sft_v1.pt`
  - Training Log: `reports/sft_training_log.json`

---

## 5. Model Evaluation & Comparison

Evaluate base versus fine-tuned models on held-out tasks:
```bash
python ml/evaluation/evaluate_models.py
```
- Evaluates validation loss, syntax validity, unit-test pass rate, and sandboxed execution.
- Generates:
  - `reports/base_model_results.json`
  - `reports/fine_tuned_model_results.json`
  - `reports/model_comparison.md`

---

## 6. Model Integration & Rollback Operations

### Inspect Registered Models:
```bash
curl http://127.0.0.1:8000/api/models
```

### Activate the Fine-Tuned Model:
```bash
curl -X POST http://127.0.0.1:8000/api/models/load -H "Content-Type: application/json" -d "{\"model_id\": \"codeforge-h-sft-v1\"}"
```

### Instant Rollback (Safe Fallback):
If unexpected model behavior occurs, execute an instant zero-downtime rollback:
```bash
curl -X POST http://127.0.0.1:8000/api/models/rollback
```
This restores the previously active model (`codeforge-mock` or `qwen3.8-27b-local`).

---

## 7. Continuous Improvement Loop

When tests or builds fail, the continuous improvement engine records sanitized failures and builds corrective training samples:
```bash
python agent/feedback/continuous_improvement.py
```
- Automatically removes paths (`C:\Users\<REDACTED>`), passwords, and API keys.
- Categorizes failure types (`model_syntax_error`, `logic_test_failure`, `dependency_missing`).
- Stages samples in `datasets/feedback_buffer/staged_corrections.jsonl`.
- Retraining is suppressed until threshold (default: 5 samples) is met to avoid churn.

---

## 8. Troubleshooting & Recovery Procedures

| Issue | Root Cause | Recovery Action |
| :--- | :--- | :--- |
| **CUDA Out Of Memory (OOM)** | Micro-batch or sequence length exceeded 4 GB VRAM | The trainer automatically catches `torch.cuda.OutOfMemoryError` and clears cache. Reduce `max_seq_len` to 256 or use CPU fallback. |
| **Corrupted Dataset File** | Incomplete download or format mismatch | Remove `datasets/raw/<file>.jsonl` and re-run `python ml/data/collection/download_datasets.py`. |
| **Model Generation Regression** | Overfitting on small dataset epoch | Trigger rollback via `/api/models/rollback`. Inspect `reports/model_comparison.md`. |
| **Subprocess Timeout in Windows** | Process hung in sandboxed execution | Sandbox uses `taskkill /F /T /PID` to cleanly terminate runaway processes after the 60s timeout ceiling. |
