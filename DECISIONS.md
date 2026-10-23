# Architecture & Engineering Decision Records (DECISIONS.md)

This log documents every architectural, algorithmic, and engineering choice made across the CodeForge AI and CodeForge-H lifecycle.

---

### DEC-001: Hardware-Aware Model Execution Tiers
- **Date:** 2026-10-23
- **Choice:** Establish three strict execution tiers:
  1. *Deterministic Mock Engine* for ultra-fast CI/CD tests and non-GPU development.
  2. *Local Ollama Engine* leveraging the user's running `http://127.0.0.1:11434` instance for natural-language agent tasks.
  3. *Small Pilot Transformer (125M-150M)* for from-scratch pretraining & fine-tuning experiments, refusing to attempt 27B pretraining locally.
- **Alternatives Considered:** Attempting to load 27B weights directly in PyTorch; using remote paid APIs without user consent.
- **Reason:** Empirical inspection revealed AMD Ryzen 5, 24 GB RAM, 6 GB VRAM RTX 3050. Loading unquantized 27B requires ~54 GB VRAM; pretraining requires 100k+ H100 hours. The chosen tiering prevents Out-Of-Memory system crashes while maximizing local capability.
- **Evidence:** `reports/hardware_report.json` and Section 4/5 of master prompt.

---

### DEC-002: Process-Level Sandbox with Timeout & Security Jail
- **Date:** 2026-10-23
- **Choice:** Implement `agent/execution/sandbox.py` using isolated subprocess execution with working-directory confinement, strict timeouts (10s default), 64 KB output truncation, and command allowlisting/denylisting.
- **Alternatives Considered:** Docker containerization.
- **Reason:** Docker engine is not installed on the host system PATH. Process-level sandboxing ensures immediate, zero-setup execution while enforcing non-destructive command policies.
- **Evidence:** Inspection showed `Get-Command docker` returned null.

---

### DEC-003: Tokenizer Strategy & Vocabulary Ceiling
- **Date:** 2026-10-23
- **Choice:** Use Byte-Level BPE with vocabulary size 49,152 (< 65,536). Shards store tokens as `uint16` to halve memmap disk footprint from 4 bytes (`uint32`) to 2 bytes per token.
- **Alternatives Considered:** SentencePiece unigram with 32k or 128k vocabulary.
- **Reason:** 49,152 provides superior code fertility (~3.6 chars/token on Python and JS/TS) while guaranteeing memory-efficient `uint16` packing.
- **Evidence:** CODEFORGE-H Section I requirements.

---

### DEC-004: Hybrid Attention / Decoder Architecture with MTP
- **Date:** 2026-10-23
- **Choice:** Modular decoder-only Transformer with causal self-attention, rotary positional embeddings (RoPE), SwiGLU non-linearities, RMSNorm, and an auxiliary Multi-Token Prediction (MTP) loss head (weight 0.2).
- **Alternatives Considered:** Standard vanilla Post-LN Transformer or pure RNN.
- **Reason:** Aligns directly with state-of-the-art coding LLMs (Qwen / DeepSeek-V3 / CodeForge-H) and enables speculative decoding verification.
- **Evidence:** CODEFORGE-H Section K & N.
