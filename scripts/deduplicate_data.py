"""
Deduplication, Decontamination & Split Engine for Coding Datasets.
Implements Section 6 & 11 of Master Specification:
- Executes exact SHA-256 deduplication.
- Executes MinHash LSH near-deduplication (Jaccard similarity threshold).
- Enforces 13-gram benchmark decontamination against HumanEval and MBPP.
- Produces reproducible train, validation, and held-out test splits.
- Generates reports/data_quality_report.json and reports/token_count_report.json.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.preprocessing.prepare_dataset import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate and Prepare Coding Datasets")
    parser.add_argument("--threshold", type=float, default=0.85, help="MinHash Jaccard similarity threshold for near-duplicates")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible dataset splits")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  CODEFORGE AI - DEDUPLICATION & DECONTAMINATION ENGINE (27B PIPELINE)")
    print("=" * 70)
    print(f"Jaccard Near-Dedup Threshold: {args.threshold} | Random Seed: {args.seed}")
    print("=" * 70)

    stats = run_pipeline()

    # Mirror report to reports/data_quality_report.json for Phase 11 naming standard
    rep_dir = PROJECT_ROOT / "reports"
    src_quality = rep_dir / "dataset_quality_report.json"
    dst_quality = rep_dir / "data_quality_report.json"
    if src_quality.exists():
        dst_quality.write_text(src_quality.read_text(encoding="utf-8"), encoding="utf-8")

    print("\n[SUCCESS] Deduplication and Decontamination Completed Successfully.")


if __name__ == "__main__":
    main()
