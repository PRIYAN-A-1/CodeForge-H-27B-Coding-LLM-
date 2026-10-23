# Dataset Legal and License Provenance Audit Report
**Generated**: 2026-10-23T07:59:25Z  
**Auditor**: CodeForge AI Automated Compliance & Provenance Engine  
**Standard**: Section 4 & Section 12 Master Specification

---

## 1. Compliance Executive Summary
All **18 candidate datasets** across pretraining, instruction tuning, reasoning, and software engineering agent trajectories were audited for:
1. License permissiveness (MIT, Apache-2.0, BSD-2/3-Clause, CC-BY-4.0, OpenRAIL-M).
2. Commercial training legality and distribution rights.
3. Mandatory attribution and provenance tracking.
4. **Benchmark Contamination Quarantine**: Isolation of test fixtures (HumanEval, MBPP) from all training splits.

---

## 2. Approved Training Datasets by Stage

### Stage 1: Source-Code Pretraining (Approved)
| Dataset | License | Commercial Usable | Attribution Required | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **The Stack v2** | `Permissive (Filtered per-file: MIT, Apache-2.0, BSD)` | Yes | Yes | Permissive open-source code |
| **The Stack v1** | `Multiple Permissive (MIT, Apache-2.0, BSD-2/3-Clause)` | Yes | No | Permissive open-source code |
| **StarCoderData** | `BigCode OpenRAIL-M v1 / Permissive per-source` | Restricted | No | Subject to BigCode OpenRAIL-M v1 acceptable use policy (no cyberattack generation or unlawful discrimination). |
| **CodeSearchNet** | `Multiple Permissive (MIT, Apache-2.0, BSD)` | Yes | No | Permissive open-source code |
| **CodeParrot GitHub Code** | `Multiple Permissive (MIT, Apache-2.0, BSD-2/3-Clause)` | Yes | No | Permissive open-source code |

### Stage 2 & 3: Supervised Instruction Tuning (Approved)
| Dataset | License | Commercial Usable | Attribution Required | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **OpenCodeInstruct** | `CC-BY-4.0` | Yes | Yes | Attribution required: Cite NVIDIA OpenCodeInstruct release and retain CC-BY-4.0 license notice. |
| **Magicoder OSS-Instruct 75K** | `Apache-2.0` | Restricted | No | Permissive SFT instructions |
| **Magicoder Evol-Instruct 110K** | `Apache-2.0` | Restricted | No | Permissive SFT instructions |
| **CodeAlpaca 20K** | `Apache-2.0` | Restricted | No | Permissive SFT instructions |
| **CodeFeedback Filtered Instruction** | `Apache-2.0` | Restricted | No | Permissive SFT instructions |
| **Glaive Code Assistant v3** | `Apache-2.0` | Restricted | No | Permissive SFT instructions |
| **APPS (Automated Programming Progress Standard)** | `MIT` | Yes | No | Permissive SFT instructions |
| **CodeXGLUE** | `MIT` | Yes | No | Permissive SFT instructions |

### Stage 4: Coding Agent Trajectories (Approved)
| Dataset | License | Commercial Usable | Attribution Required | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **SWE-bench** | `MIT` | Yes | No | Verified tool interactions |
| **SWE-smith** | `MIT` | Yes | No | Verified tool interactions |
| **OpenHands Agent Trajectories** | `MIT` | Yes | No | Verified tool interactions |

---

## 3. Quarantined Benchmarks (Contamination Prevention)
The following datasets are **STRICTLY QUARANTINED** from all training pipelines. They are reserved exclusively for independent held-out evaluation to ensure measurement integrity:

| Benchmark | License | Quarantine Status | Rational |
| :--- | :--- | :--- | :--- |
| **HumanEval** | `MIT` | **QUARANTINED** | Held-out benchmark; strictly prohibited from training data to avoid test contamination. |
| **MBPP (Mostly Basic Python Problems)** | `CC-BY-4.0` | **QUARANTINED** | Held-out benchmark; strictly prohibited from training data to avoid test contamination. |

---

## 4. Mandatory Legal and Attribution Requirements
- **StarCoderData: Adherence to OpenRAIL-M behavioral restrictions required.**
- **OpenCodeInstruct: Attribution to NVIDIA is legally mandatory under CC-BY-4.0.**

---
## 5. Verification Protocol
All ingested files must pass SHA-256 integrity verification, PII/secret scrubbing, and duplicate suppression prior to inclusion in processed training splits.
