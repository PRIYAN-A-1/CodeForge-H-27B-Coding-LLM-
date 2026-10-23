# CodeForge AI & CodeForge-H

> **AI-Powered Multilingual Coding Model & Autonomous Software Development Agent**

[![Tests](https://img.shields.io/badge/pytest-27%20passed%20(100%25)-emerald)](file:///g:/My%20Drive/thoooo/tests)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](file:///g:/My%20Drive/thoooo/LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue)](file:///g:/My%20Drive/thoooo/ENVIRONMENT.md)
[![Hardware](https://img.shields.io/badge/GPU-RTX%203050%20(6GB)-purple)](file:///g:/My%20Drive/thoooo/reports/hardware_report.json)
[![Model](https://img.shields.io/badge/CodeForge--H-126.9M%20params-orange)](file:///g:/My%20Drive/thoooo/model/hybrid_transformer.py)

---

## 1. Executive Summary

CodeForge AI converts natural language requirements into fully structured, tested, and runnable software projects. Rather than generating disconnected snippets or static text, CodeForge AI executes a bounded engineering loop:

1. **Requirement Planning:** Validates requirements into structured Pydantic `ProjectPlan` schemas, selects optimal tech stacks, and flags ambiguity.
2. **Multi-File Project Synthesis:** Generates directory structures, configurations, schemas, components, and unit tests with directory traversal protection.
3. **Sandboxed Verification:** Executes tests (`pytest`, syntax verification) within an isolated workspace jail.
4. **Bounded Debugging & Repair:** Analyzes compiler and test failures, applies minimal patches, and reruns tests within a strict retry budget (default 3 attempts).
5. **ZIP Export:** Packages the completed project for one-click download.

---

## 2. Three Operating Modes

| Mode | Identity | Purpose | Local Feasibility |
| :--- | :--- | :--- | :--- |
| **Mode A** | **Existing Model Inference** | Local deterministic mock engine, Ollama (`qwen3.8:27b`), and Hugging Face Transformers. | **Feasible** (Instant mock / Ollama) |
| **Mode B** | **Supervised Fine-Tuning (SFT)** | Parameter-efficient LoRA/QLoRA adapter training with assistant label masking. | **Feasible** (Smoke test verified) |
| **Mode C** | **Foundation Model Pretraining** | CodeForge-H Hybrid Architecture (3x [3 DeltaNet + 1 Attn], RoPE, SwiGLU, RMSNorm, tied embeddings, MTP head: 126.94M params). | **Feasible** (Pilot smoke test verified) |

> **Critical Safety Constraint:** Unquantized 27B pretraining is strictly blocked on local workstations to prevent system crashes (`scripts/estimate_cost.py`).

---

## 3. Verified System Hardware & Telemetry

Empirical inspection of the host system:
- **CPU:** AMD Ryzen 5 7235HS (4 physical cores, 8 logical threads)
- **RAM:** 23.69 GB physical RAM (~4.2 GB available at idle)
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, Driver 591.74, CUDA 13.1)
- **Local Inference Server:** Ollama running at `http://127.0.0.1:11434` with `qwen3.8:27b` (17 GB) detected
- Detailed reports: [hardware_report.json](file:///g:/My%20Drive/thoooo/reports/hardware_report.json) and [resource_plan.md](file:///g:/My%20Drive/thoooo/reports/resource_plan.md).

---

## 4. Quickstart Guide

### 4.1 Start Backend API & Interactive Dashboard
```powershell
# In repository root
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive UI Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **FastAPI Interactive Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4.2 Autonomous End-to-End Execution Pipeline
```powershell
python -m serve.pipeline "Build an expense management system with Next.js and FastAPI"
```

### 4.3 Unified Developer CLI (`codeforge`)
```powershell
python -m cli.codeforge plan "Build a real-time chat app"
python -m cli.codeforge gen "Build an invoice tracker with Next.js"
python -m cli.codeforge models
python -m cli.codeforge eval
python -m cli.codeforge hardware
```

### 4.4 Run Full Test Suite
```powershell
pytest -v tests/
```
**Test Results (27/27 Passed, 100% Green):**
- `tests/e2e/test_e2e_workflow.py`: 1 passed (complete autonomous lifecycle)
- `tests/integration/test_integration_api.py`: 6 passed (REST API & streaming)
- `tests/security/test_security_sandbox.py`: 5 passed (jail, timeout, traversal)
- `tests/test_model_section_k.py`: 3 passed (DeltaNet reference, causal invariance, finite gradients)
- `tests/unit/test_cli_pipeline.py`: 4 passed (CLI commands, pipeline run, zip export)
- `tests/unit/test_unit_core.py`: 8 passed (tokenizer, BPE, FIM, dedup, config)

---

## 5. Repository Structure

```
codeforge-ai/
├── apps/
│   ├── web/                     # Next.js 15 + Tailwind CSS + TypeScript frontend
│   └── api/                     # FastAPI backend server & static dashboard UI
├── packages/
│   └── shared/                  # Pydantic v2 schemas and type definitions
├── ml/
│   ├── data/
│   │   ├── preprocessing/       # Cleaning, heuristic filters, deduplication
│   │   ├── tokenization/        # BPE tokenizer, FIM pipeline, uint16 packing
│   │   └── validation/          # Manifest validation & 13-gram decontamination
│   ├── training/
│   │   ├── pretraining/         # From-scratch CodeForge-H model & trainer
│   │   └── finetuning/          # SFT LoRA pipeline & adapter checkpointing
│   ├── evaluation/              # Unbiased pass@k & benchmark fixtures
│   ├── inference/               # Inference engine (Mock, Ollama, Transformers)
│   └── model_registry/          # Capability catalog & VRAM preflight checks
├── agent/
│   ├── planner/                 # Requirement parser & architecture planner
│   ├── generator/               # Multi-file project code generator
│   ├── execution/               # Process-bounded sandbox jail & timeouts
│   └── repair/                  # Bounded debugging & repair engine
├── datasets/
│   ├── manifests/               # Dataset provenance & legal tracking
│   ├── raw/                     # Raw datasets (excluded from git)
│   └── processed/               # Tokenized memmap shards
├── tests/
│   ├── unit/                    # Unit tests for cleaning, tokenization, config
│   ├── integration/             # API endpoint integration tests
│   ├── security/                # Sandboxing, injection, and path traversal tests
│   └── e2e/                     # End-to-end requirement to ZIP export tests
├── scripts/
│   └── estimate_cost.py         # FLOPs, compute ladder & budget gatekeeper
├── configs/
│   ├── budget.yaml              # Compute budget ceiling ($300 max)
│   └── config.py                # Pydantic application settings
├── reports/
│   ├── hardware_report.json     # Empirical host hardware profile
│   ├── resource_plan.md         # Compute tier feasibility plan
│   ├── decontam_report.json     # 13-gram benchmark decontamination audit
│   └── dataset_quality_report.json # Dataset quality and deduplication metrics
├── docs/                        # Complete technical documentation suite
├── .env.example                 # Environment configuration template
├── .gitignore                   # Exclusions for weights, databases, node_modules
├── docker-compose.yml           # Multi-container production deployment
├── Dockerfile                   # Backend container definition
├── DECISIONS.md                 # Architecture decision records
├── RISKS.md                     # Risk management register
├── MODEL_CARD.md                # CodeForge-H model specification & targets
├── DATA_CARD.md                 # Data sources, licensing & governance
└── ENVIRONMENT.md               # Runtime environment specifications
```
