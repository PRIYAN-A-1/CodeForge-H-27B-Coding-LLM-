# Data Card: CodeForge AI & CodeForge-H

## Overview
This document catalogs all datasets, mixtures, licenses, and preprocessing filters utilized for training and evaluating CodeForge-H models.

## 1. Permitted Data Sources & Licensing

| Source Name | Category | Primary License | Gating / Terms | Pilot Token Allocation | Cleaned / Decontaminated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StarCoderData / The Stack v2 (Filtered)** | Permissive Code (Py, JS, TS, SQL) | MIT, Apache-2.0, BSD-2/3, ISC | Permissive allowlist, opt-out honored | 12.0B tokens (60%) | Yes (13-gram scrubbed) |
| **Synthetic Verified Pairs** | Natural Language → Spec → Code + Tests | Generated & Verified Apache-2.0 | Sandbox verified (pytest green) | 0.5B tokens (2.5%) | Yes |
| **Curated Documentation (Python/MDN)** | Technical Docs & APIs | CC-BY / PSF / W3C | Non-commercial restriction filtered out | 3.0B tokens (15%) | Yes |
| **Technical Q&A / Issues** | Multi-Turn Problem Solving | CC BY-SA (with attribution) | Attribution metadata preserved | 1.5B tokens (7.5%) | Yes |
| **STEM & Code Math (OpenWebMath subset)** | Algorithmic reasoning | Permissive Web | Filtered via fastText / heuristics | 1.0B tokens (5%) | Yes |
| **CommitPackFT / Multi-File Diffs** | Code review & debugging | MIT / Apache-2.0 | Git commit history | 2.0B tokens (10%) | Yes |

## 2. Preprocessing & Quality Filtering Rules
1. **Language & Extension Filter:** Strictly parse code files matching target languages using Linguist rules.
2. **License Guardrail:** Discard any file containing non-permissive or copyleft licenses (GPL, AGPL, SSPL, or unknown licenses).
3. **Length & Quality Heuristics:**
   - Drop any document where `max_line_length > 1000` or `mean_line_length > 100`.
   - Drop any document where alphanumeric character fraction < 0.25.
   - Drop minified files, lockfiles, base64 blobs, or files > 1 MB or < 50 Bytes.
4. **Exact & Near-Deduplication:**
   - Exact deduplication via SHA-256 of whitespace-normalized text.
   - Near-deduplication using MinHash (128 hash permutations, 5-gram shingles, Jaccard threshold ~0.80).
5. **PII & Secret Redaction:**
   - Regex patterns for API keys, bearer tokens, private keys, IPv4/v6 addresses, and email addresses. Replaced with `<KEY>`, `<EMAIL>`, `<IP>`, `<NAME>`.
6. **Decontamination Protocol:**
   - All documents are screened against 13-gram shingles of HumanEval, MBPP, EvalPlus, and MultiPL-E prompts AND reference solutions. Any matching document is discarded.
