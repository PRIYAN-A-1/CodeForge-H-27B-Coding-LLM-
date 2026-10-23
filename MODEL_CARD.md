# Model Card: CodeForge-H & CodeForge AI

## Model Details
- **Model Name:** CodeForge-H (Hybrid Reasoning Code Language Model)
- **Architecture:** Decoder-Only Causal Transformer with Rotary Position Embeddings (RoPE), SwiGLU FFN, RMSNorm, and Multi-Token Prediction (MTP) auxiliary objective.
- **Pilot Parameter Size:** 125M – 150M parameters (d_model=768, 12 layers, 12 heads, FFN=2048, vocab=49,152)
- **Target Parameter Scaling:** 350M (S1), 1B (M1), 4B (L1), 27B (XL - cloud tier)
- **Context Length:** 2,048 tokens baseline, extendable up to 32,768 tokens via RoPE base scaling
- **Primary Languages:** Python, JavaScript/TypeScript, SQL, HTML/CSS
- **License:** Apache-2.0

## Intended Use
CodeForge-H is designed to convert plain-English natural language requirements into fully structured, tested, and runnable multi-file software projects. It supports two execution modes:
1. **Reasoning Mode (`<think> ... </think>`):** Generates architectural planning and reasoning trace prior to producing code.
2. **Direct Mode:** Generates direct structured code with minimal latency.

## Target & Measured Benchmark Metrics

| Metric | Target (Pilot 140M) | Target (M1 1B) | Target (XL 27B) | Measured Baseline (Pilot Smoke) |
| :--- | :--- | :--- | :--- | :--- |
| **Private Requirements Pass Rate** | ≥ 15.0% | ≥ 45.0% | ≥ 78.0% | Recorded in Eval Phase |
| **HumanEval+ Pass@1** | 8.0 – 14.0% | 22.0 – 30.0% | 55.0 – 68.0% | Recorded in Eval Phase |
| **MBPP+ Pass@1** | 10.0 – 16.0% | 25.0 – 35.0% | 60.0 – 72.0% | Recorded in Eval Phase |
| **Pass@10 (T=0.8, n=20)** | ≥ 22.0% | ≥ 45.0% | ≥ 82.0% | Recorded in Eval Phase |
| **Long Context (32k Needle)** | ≥ 95.0% | ≥ 98.0% | ≥ 99.5% | Recorded in Eval Phase |
| **p50 Latency (tokens/sec)** | ≥ 85 t/s (CPU) | ≥ 40 t/s (GPU) | ≥ 25 t/s (Quantized) | Tested & Logged |
| **Security Findings / 1k LOC** | < 1.0 | < 0.5 | < 0.1 | Tested via Bandit / AST |

## Tokenizer & Vocabulary
- **Algorithm:** Byte-Level BPE (ByteLevelBPETokenizer)
- **Vocabulary Size:** 49,152 (indexed in `uint16` memmap arrays)
- **Special Tokens:** `<|endoftext|>`, `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`, `<fim_pad>`, `<filename>`, `<reponame>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_result>`, `</tool_result>`

## Training Data & Governance
Trained exclusively on permissively licensed code (MIT, Apache-2.0, BSD-2/3, ISC), synthetic requirement-to-code pairs verified via isolated test execution, and curated documentation. Benchmarks (HumanEval, MBPP, EvalPlus) are decontaminated using strict 13-gram exact match filtering.
