# CodeForge AI — Data-Driven Fine-Tuning Execution Plan

**Objective:** Autonomous acquisition, preparation, fine-tuning, evaluation, and integration of domain-specific coding data to enhance the existing CodeForge AI agent.

---

## 1. Targeted Capabilities & Weakness Matrix

The fine-tuning dataset is engineered to address high-friction software agent tasks:
1. **Multi-File Project Architecture:** Coherent code splitting across `backend/main.py`, `backend/requirements.txt`, `backend/test_main.py`, and `frontend/package.json`.
2. **Schema & Contract Consistency:** Strict Pydantic v2 schemas and FastAPI status codes (HTTP 200/201/404/422).
3. **Automated Unit Testing:** Writing complete, non-stub Pytest functions using `TestClient` and testing both happy and error paths.
4. **Self-Repair & Debugging:** Ingesting tracebacks (e.g. `AttributeError`, `ImportError`, `TimeoutExpired`), identifying the faulty line, and outputting minimal corrective diffs.
5. **Multilingual Syntax:** Python, TypeScript, SQL, and Shell scripting.

---

## 2. Dataset Discovery, Licensing & Acquisition

### Candidate Source Evaluation:

| Dataset | Primary Source | License | Suitability | Acquisition Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Magicoder OSS-Instruct** | `ise-uiuc/Magicoder-OSS-Instruct-75K` | MIT | **Approved (Training)** | Stream bounded 2,000-sample subset focusing on multi-lingual problem solving. |
| **CodeFeedback Filtered** | `m-a-p/CodeFeedback-Filtered-Instruction` | Apache-2.0 | **Approved (Repair & Feedback)** | Extract targeted error-correction and code-refinement dialogs. |
| **The Stack v2 (Subset)** | `bigcode/the-stack-v2` | Permissive (MIT/Apache) | **Approved (Source Code)** | High-quality Python and TypeScript multi-file structures. |
| **MBPP** | Google Research / Hugging Face | CC-BY-4.0 | **Approved (Evaluation Only)** | Strictly reserved for held-out evaluation; excluded from training splits. |
| **Synthetic Agent Verification Pairs** | Local Sandbox Engine | Apache-2.0 | **Approved (Core Agent Alignment)** | Pytest-verified input-output pairs matching CodeForge agent workflows. |

### Size & Memory Guardrails:
* Downloads are streamed with sample bounds (total raw data $\le 250$ MB).
* Memory-mapped streaming prevents exceeding the 16 GB host RAM limit.
* Full license provenance and attribution tracked in `reports/dataset_manifest.json`.

---

## 3. Preprocessing, Tokenization & Decontamination Pipeline

```
Raw Data Ingestion
       │
       ▼
[clean_dataset.py]  ──> Heuristic filters (line lengths, code/comment ratio)
       │            ──> PII redaction (<KEY>, <EMAIL>, <IP>)
       ▼
[deduplicate.py]    ──> Exact SHA-256 hash deduplication
       │            ──> MinHash LSH near-duplicate removal (threshold 0.85)
       ▼
[validate_manifest] ──> 13-gram benchmark decontamination against HumanEval & MBPP
       │
       ▼
[Chat Formatting]   ──> <|im_start|>system...<|im_start|>user...<|im_start|>assistant...
       │
       ▼
[Token Packing]     ──> ByteLevel BPE tokenization + Assistant-only loss masking
       │
       ▼
[Train/Val Splits]  ──> datasets/processed/train.jsonl (90%)
                    ──> datasets/processed/validation.jsonl (10%)
```

---

## 4. Training Hyperparameters (LoRA on 4 GB VRAM)

| Parameter | Configuration | Technical Rationale |
| :--- | :--- | :--- |
| **Model Target** | `CodeForge-H Pilot` / `Qwen2.5-Coder-0.5B` | Sized strictly for $\le 4.0$ GB VRAM footprint |
| **Adapter Rank ($r$)** | 8 | Sufficient capacity for syntax and agent formatting |
| **Adapter Alpha ($\alpha$)** | 16 | Standard $2 \times r$ scaling factor |
| **LoRA Target Modules** | `q_proj`, `v_proj`, `out_proj` | Balances attention adaptation with minimal parameter footprint |
| **Micro-Batch Size** | 1 | Eliminates risk of CUDA Out-Of-Memory spikes |
| **Gradient Accumulation** | 4 steps | Simulates effective batch size of 4 |
| **Learning Rate** | $2.0 \times 10^{-4}$ | Standard AdamW rate for LoRA fine-tuning |
| **LR Schedule** | Cosine with 10% linear warmup | Smooth convergence without early gradient shock |
| **Weight Decay** | 0.01 | Defensive regularization against overfitting on small corpus |
| **Max Sequence Length** | 512 tokens | Captures full multi-file functions while bounding activation memory |
| **Loss Masking** | Assistant-Only (`label = -100` on prompt) | Focuses gradient updates strictly on model-generated code |

---

## 5. Evaluation, Acceptance & Regression Gates

Prior to promotion, the fine-tuned checkpoint must pass:
1. **Perplexity / Validation Loss:** Validation loss must decrease monotonically compared to step 0.
2. **Benchmark Pass@1:** Score on held-out Python/TypeScript functional tests must meet or exceed baseline.
3. **Agent Integration Regression:** All 27 existing test suites in `tests/` must remain **100% green**.
4. **Live Project Generation:** Autonomous pipeline must produce a valid 9-file project with 100% passing tests in under 30 seconds.
5. **Zero Host Vulnerability:** All generated code runs inside the `ExecutionSandbox` with directory traversal protection.
