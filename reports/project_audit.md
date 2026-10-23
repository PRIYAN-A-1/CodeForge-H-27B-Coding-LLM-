# CodeForge AI — Project Audit Report

**Date:** October 2026  
**Auditor:** Autonomous AI Research & Systems Engineer  
**Branch:** `feature/autonomous-data-training-integration`  
**Git Baseline Commit:** `ea7886d`  

---

## 1. Executive Summary

This audit assesses the active state of the **CodeForge AI** project prior to autonomous dataset acquisition, model fine-tuning, and integration. The system is a fully operational, test-verified coding agent and hybrid language model platform.

All existing subsystems are intact, and all 27 unit, integration, security, architecture, and end-to-end tests are passing with 100% success.

---

## 2. Subsystem & Component Inventory

### 2.1 Agent Core (`agent/`)
* **`agent/planner/planner.py`**: Decomposes natural language requirements into structured Pydantic v2 `ProjectPlan` schemas. Identifies project type, selects frontend/backend/database stacks, extracts features, and formulates clarification questions.
* **`agent/generator/generator.py`**: Synthesizes production multi-file projects (FastAPI REST endpoints, Next.js UI, Pydantic schemas, unit tests, requirements.txt, Dockerfile). Enforces strict directory traversal validation (`../` prevention).
* **`agent/execution/sandbox.py`**: Process-isolated sandbox jail. Blocks forbidden commands (`rm -rf`, `format`, fork-bombs), enforces a 64 KB stdout/stderr buffer ceiling, 60-second execution timeouts, and kills Windows process trees (`taskkill /F /T /PID`).
* **`agent/repair/repair.py`**: Bounded self-repair loop (default 3 retries). Automatically captures syntax and pytest failures, synthesizes targeted patches, applies them in the isolated workspace, and re-runs tests.

### 2.2 Serving, CLI & Interfaces (`serve/`, `cli/`, `apps/`, `ide/`)
* **`serve/pipeline.py`**: Autonomous end-to-end execution loop: NL Requirement $\to$ Chain-of-Thought Trace $\to$ Plan $\to$ Code Synthesis $\to$ Sandbox Tests $\to$ Bounded Self-Repair $\to$ Deployable ZIP bundle.
* **`cli/codeforge.py`**: Developer CLI providing `plan`, `gen`, `test`, `repair`, `models`, `hardware`, `eval`, `pretrain`, `sft`, and `serve`.
* **`apps/api/main.py`**: FastAPI backend server running on `http://127.0.0.1:8000`. Exposes `/api/health`, `/api/models`, `/api/projects/*`, `/api/usage/tokens`.
* **`apps/api/static/index.html`**: Zero-dependency dark-mode developer dashboard mounted at `GET /` and `GET /dashboard`.
* **`apps/web/`**: Full Next.js 15 + TypeScript + Tailwind CSS application codebase.
* **`ide/vscode_extension/`**: VS Code extension manifest (`package.json`) and runtime script (`extension.js`).

### 2.3 Model & Training Subsystems (`model/`, `ml/`)
* **`model/deltanet_reference.py`**: Recurrent Gated DeltaNet linear attention formulation:
  $$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
* **`model/hybrid_transformer.py`**: CodeForge-H hybrid architecture:
  * 3 macro-blocks of [3 Gated DeltaNet layers + 1 Multi-Head Attention layer] (12 total layers: 9 DeltaNet + 3 Full Attention).
  * RoPE rotary embeddings, GQA (12 Q-heads, 4 KV-heads), SwiGLU FFN ($d_{\text{ffn}}=1536$), RMSNorm, tied embeddings.
  * Multi-Token Prediction (MTP) projection head ($2 \times d_{\text{model}} \to d_{\text{model}}$).
  * Exact parameter count: **126,941,952 parameters (126.94M)**.
* **`ml/training/pretraining/train.py`**: AdamW foundation pretrainer with cosine LR and MTP loss.
* **`ml/training/finetuning/sft.py`**: LoRA Supervised Fine-Tuning trainer with assistant-only loss masking.
* **`ml/inference/engine.py`**: Inference dispatcher supporting `mock`, `ollama` (`qwen3.8:27b`), and `transformers`.
* **`ml/model_registry/registry.py`**: Dynamic model catalog and VRAM preflight checks.
* **`ml/evaluation/eval_harness.py`**: Unbiased pass@k estimator, reference baseline (100%), and stub verification (0%).
* **`ml/data/`**: Preprocessing (`clean_dataset.py`, `deduplicate.py`), tokenization (`tokenizer.py`, `fim_pipeline.py`), and validation (`validate_manifest.py`).

---

## 3. External Dataset Repository Audit

The user provided URL was empirically inspected:
* **Target:** `https://github.com/PRIYAN-A-1/CODING-AGENT-CODING-DATASETV2`
* **HTTP Status:** `404 Not Found`
* **Result:** The repository is either private, requires GitHub authentication credentials not present in the environment, or does not exist at this path.
* **Action:** Per instructions, its contents will not be fabricated. Publicly accessible and approved coding instruction datasets (Magicoder OSS-Instruct, CodeFeedback, The Stack v2 permissive subset, MBPP) will be used with verified license compliance.

---

## 4. Hardware Verification & Constraints

| Resource | Host Hardware | Conservative Allocation |
| :--- | :--- | :--- |
| **GPU VRAM** | NVIDIA GeForce RTX 3050 Laptop (6,144 MB total) | **Capped at 4.0 GB VRAM** (strict low-budget requirement) |
| **System RAM** | 23.69 GB physical (~4.2 GB free) | **Streaming / Chunked processing** to fit 16 GB baseline |
| **CUDA / Torch** | NVIDIA Driver 591.74, CUDA 13.1 | Python 3.13 default environment uses CPU PyTorch; GPU operations utilize memory-efficient QLoRA or quantized inference |
| **Storage** | 63.35 GB free on workspace drive | Bounded dataset streaming, max 250 MB raw data downloads |

---

## 5. Integration Plan for Trained Model

The fine-tuned model / adapter will be integrated as follows:
1. **Checkpointing:** Weights exported to `ml/training/checkpoints/codeforge_lora_sft_v1.pt`.
2. **Registry:** Registered in `ml/model_registry/registry.py` under model ID `codeforge-h-sft-v1`.
3. **Inference Dispatch:** Loaded via `ml/inference/engine.py` with fallback to `codeforge-mock` and `ollama` (`qwen3.8:27b`).
4. **Agent Invocation:** `agent/generator/generator.py` and `serve/pipeline.py` will use the registered fine-tuned model when active.
5. **Rollback & Fallback:** Original baseline model and mock engine preserved as zero-latency failover.
