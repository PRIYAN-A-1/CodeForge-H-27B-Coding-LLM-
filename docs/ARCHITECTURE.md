# CodeForge AI — Architecture Specification

## 1. System Overview
CodeForge AI is a modular, hardware-aware autonomous software engineering agent and multilingual coding language model. It takes high-level natural language requirements and autonomously produces fully structured, tested, and runnable multi-file software projects.

```
                      +-----------------------------+
                      |  Next.js Frontend (Web App) |
                      +--------------+--------------+
                                     |
                           HTTP REST / SSE Stream
                                     |
                      +--------------v--------------+
                      |     FastAPI Backend Core    |
                      +-------+-------------+-------+
                              |             |
           +------------------+             +------------------+
           |                                                   |
+----------v-----------+                            +----------v-----------+
|    Agent Engine      |                            |    ML Subsystem      |
| - Requirement Planner|                            | - Model Registry     |
| - Code Generator     |                            | - Inference Engine   |
| - Bounded Repair     |                            | - Token Accounting   |
| - Sandbox Jail       |                            | - From-Scratch Model |
+----------+-----------+                            | - LoRA SFT Pipeline  |
           |                                        +----------------------+
+----------v-----------+
| Isolated Workspace   |
| sandbox/projects/    |
+----------------------+
```

## 2. Core Modules

### 2.1 Requirement Planner (`agent/planner/planner.py`)
Parses natural language requirements into strongly-typed Pydantic schemas (`ProjectPlan`). Identifies implicit stack requirements, security baselines, and generates clarifying questions.

### 2.2 Code Generator (`agent/generator/generator.py`)
Generates comprehensive multi-file project workspaces. Prevents directory traversal attacks by validating all relative paths against base project directories. Writes complete, non-placeholder files for backend, frontend, unit tests, and configs.

### 2.3 Sandbox Execution Jail (`agent/execution/sandbox.py`)
Executes validation and testing commands strictly inside project workspaces. Enforces command safety allowlists, timeouts (10s default), 64 KB output buffer ceilings, and kills process trees cleanly upon timeout.

### 2.4 Bounded Repair Engine (`agent/repair/repair.py`)
Analyzes compiler errors and pytest failures, generates targeted patches, and reruns tests within a strict retry budget (default 3 attempts).

### 2.5 ML Architecture (`ml/training/pretraining/model.py`)
Implements CodeForge-H: a decoder-only causal Transformer with:
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA) with 12 Q heads and 4 KV heads
- SwiGLU non-linearities without bias terms
- RMSNorm pre-normalization
- Multi-Token Prediction (MTP) auxiliary objective predicting token $t+2$ with loss weight 0.2
