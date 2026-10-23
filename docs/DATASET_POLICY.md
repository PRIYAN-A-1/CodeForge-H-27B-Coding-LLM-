# CodeForge AI — Dataset & Governance Policy (DATASET_POLICY.md)

## 1. Legal and Compliance Framework
CodeForge AI adheres strictly to the following principles:
- **Permissive-Only Ingestion:** Only source code published under permissive open-source licenses (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, PSF) is eligible for inclusion in pretraining and fine-tuning mixtures.
- **Copyleft Exclusion:** Restrictive licenses (GPL, AGPL, SSPL) and proprietary repositories are strictly rejected by the license classifier (`validate_license`).
- **Opt-Out Compliance:** Respects repository-level opt-out manifests (`opt-out-v2`).
- **Attribution Metadata:** Retains repository name, file path, commit hash, and license attribution for every ingested sample.

## 2. Preprocessing & Quality Filters
Every ingested document passes through our deterministic pipeline (`ml/data/preprocessing/clean_dataset.py`):
1. **Length Bounds:** Discards files with maximum line length > 1,000 characters or mean line length > 100 characters.
2. **Size Bounds:** Only files between 50 Bytes and 1 MB are retained.
3. **Alphanumeric Ratio:** Requires an alphanumeric character ratio >= 0.25 (drops binary dumps, minified scripts, and base64 blobs).
4. **Exact & Near Deduplication:** SHA-256 whitespace-normalized exact deduplication, followed by MinHash near-deduplication (Jaccard similarity threshold >= 0.80).

## 3. PII and Secret Redaction
All source files are scanned for secrets before tokenization:
- API keys, access tokens, and passwords -> `<KEY>`
- Email addresses -> `<EMAIL>`
- IPv4/v6 addresses -> `<IP>`

## 4. Benchmark Decontamination Protocol
To guarantee empirical evaluation validity:
- Every document is scanned against 13-gram shingles of HumanEval, MBPP, EvalPlus, and MultiPL-E prompts and canonical solutions.
- Any document exhibiting a 13-gram exact match is discarded immediately and logged in `reports/decontam_report.json`.
- Zero contaminated samples are permitted in training splits.
