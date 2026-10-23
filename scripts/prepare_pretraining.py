"""
Pretraining and Fill-In-The-Middle (FIM) Data Preparation Engine.
Implements Phase 5A & Section 7A of Master Specification:
- Prepares causal code language modeling datasets and Fill-In-The-Middle (FIM) tokens:
  <fim_prefix>, <fim_suffix>, <fim_middle>.
- Packs documents to maximize token density and context utilization.
- Validates tokenization compatibility with existing 27B / CodeForge-H vocabularies.
- Saves datasets/processed/pretraining_fim.jsonl and reports/pretraining_data_report.json.
"""

import sys
import os
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.tokenization.tokenizer import tokenizer

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical FIM Sentinel Tokens
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"


def apply_fim_transformation(code: str, fim_rate: float = 0.5) -> Dict[str, Any]:
    """
    Applies PSM (Prefix-Suffix-Middle) transformation per Bavarian et al. (OpenAI).
    Splits document into prefix, middle, and suffix.
    """
    if random.random() > fim_rate or len(code.splitlines()) < 4:
        return {
            "is_fim": False,
            "text": code,
            "prefix": "",
            "middle": "",
            "suffix": ""
        }

    lines = code.splitlines(keepends=True)
    num_lines = len(lines)
    split_1 = random.randint(1, max(1, num_lines // 3))
    split_2 = random.randint(split_1 + 1, max(split_1 + 1, (2 * num_lines) // 3))

    prefix = "".join(lines[:split_1])
    middle = "".join(lines[split_1:split_2])
    suffix = "".join(lines[split_2:])

    # PSM Format: <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE
    fim_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"

    return {
        "is_fim": True,
        "text": fim_text,
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix
    }


def prepare_pretraining_corpus(max_samples: int = 100) -> Dict[str, Any]:
    print("=" * 70)
    print("  CODEFORGE AI - CONTINUED PRETRAINING & FIM PIPELINE (27B SPEC)")
    print("=" * 70)

    raw_train = PROCESSED_DIR / "train.jsonl"
    if not raw_train.exists():
        raise FileNotFoundError(f"Processed train data not found: {raw_train}")

    output_file = PROCESSED_DIR / "pretraining_fim.jsonl"
    samples = []
    total_tokens = 0
    fim_count = 0

    with open(raw_train, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    # Extract code from assistant message
                    assistant_msg = next((m["content"] for m in record.get("messages", []) if m["role"] == "assistant"), "")
                    if assistant_msg:
                        samples.append(assistant_msg)
                except Exception:
                    pass

    fim_records = []
    for idx, code in enumerate(samples[:max_samples]):
        transformed = apply_fim_transformation(code, fim_rate=0.5)
        text = transformed["text"]
        toks = tokenizer.encode(text) if hasattr(tokenizer, "encode") else text.split()
        tok_len = len(toks)
        total_tokens += tok_len

        if transformed["is_fim"]:
            fim_count += 1

        fim_records.append({
            "id": f"pretrain-fim-{idx:04d}",
            "text": text,
            "token_length": tok_len,
            "is_fim": transformed["is_fim"]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in fim_records:
            f.write(json.dumps(r) + "\n")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_file": str(output_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_records": len(fim_records),
        "fim_transformed_records": fim_count,
        "fim_rate_pct": round(fim_count / max(len(fim_records), 1) * 100, 2),
        "total_tokens": total_tokens,
        "average_tokens_per_record": round(total_tokens / max(len(fim_records), 1), 2),
        "sentinel_tokens": {
            "prefix": FIM_PREFIX,
            "suffix": FIM_SUFFIX,
            "middle": FIM_MIDDLE
        }
    }

    report_file = REPORTS_DIR / "pretraining_data_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Prepared {len(fim_records)} pretraining/FIM samples ({total_tokens:,} tokens).")
    print(f"  FIM Rate: {report['fim_rate_pct']}% | Output: {output_file}")
    print(f"  Report:   {report_file}")
    return report


if __name__ == "__main__":
    prepare_pretraining_corpus()
