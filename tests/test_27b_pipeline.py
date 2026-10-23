"""
Automated Unit and Integration Tests for 27B Coding LLM Pipeline.
Verifies dataset registry, legal auditing, FIM pretraining, SFT loss masking,
agent trajectories, 27B memory calculations, and export bundles.
"""

import sys
import json
import csv
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_licenses import LicenseValidator
from scripts.prepare_pretraining import apply_fim_transformation, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE
from scripts.prepare_sft import format_chatml_example


def test_dataset_registry_json_and_csv_exist_and_valid():
    json_path = PROJECT_ROOT / "datasets" / "registry.json"
    csv_path = PROJECT_ROOT / "datasets" / "registry.csv"

    assert json_path.exists(), "datasets/registry.json missing"
    assert csv_path.exists(), "datasets/registry.csv missing"

    with open(json_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert len(registry) >= 15, f"Expected at least 15 candidates, found {len(registry)}"

    # Check required fields
    required_keys = [
        "dataset_name", "canonical_url", "publisher", "category",
        "license", "commercial_use_restrictions", "data_format",
        "recommended_training_stage"
    ]
    for entry in registry:
        for k in required_keys:
            assert k in entry, f"Missing key {k} in {entry.get('dataset_name')}"

    # Verify CSV line count matches JSON
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == len(registry)


def test_license_validation_quarantines_benchmarks():
    validator = LicenseValidator()
    audit = validator.audit_licenses()

    assert audit["summary_counts"]["approved_for_training"] >= 10
    assert audit["summary_counts"]["quarantined_for_evaluation"] >= 2

    quarantined_names = [d["dataset_name"] for d in audit["evaluation_quarantined_benchmarks"]]
    assert "HumanEval" in quarantined_names
    assert "MBPP (Mostly Basic Python Problems)" in quarantined_names

    # Check mandatory legal warnings
    assert len(audit["mandatory_legal_notices"]) >= 1
    assert any("OpenCodeInstruct" in w for w in audit["mandatory_legal_notices"])


def test_27b_memory_calculation_math():
    sizing_json = PROJECT_ROOT / "reports" / "27b_hardware_memory_sizing.json"
    assert sizing_json.exists()

    with open(sizing_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = data["configurations"]
    assert configs["full_finetuning_fp32"]["total_vram_required_gb"] == 480.0
    assert configs["full_finetuning_bf16"]["total_vram_required_gb"] == 244.0
    assert configs["qlora_int4_rank16"]["total_vram_required_gb"] < 24.0
    assert configs["qlora_int4_rank16"]["feasible_on_local_rtx3050"] is False


def test_pretraining_fim_transformation():
    code = (
        "def compute_interest(principal, rate, years):\n"
        "    interest = principal * rate * years\n"
        "    total = principal + interest\n"
        "    return round(total, 2)\n"
    )
    res = apply_fim_transformation(code, fim_rate=1.0)
    assert res["is_fim"] is True
    assert FIM_PREFIX in res["text"]
    assert FIM_SUFFIX in res["text"]
    assert FIM_MIDDLE in res["text"]


def test_sft_chatml_assistant_loss_masking():
    messages = [
        {"role": "system", "content": "You are a code agent."},
        {"role": "user", "content": "Write hello world in python"},
        {"role": "assistant", "content": "print('hello world')"}
    ]
    formatted = format_chatml_example(messages)
    input_ids = formatted["input_ids"]
    labels = formatted["labels"]

    assert len(input_ids) == len(labels)
    # The prompt portion must be masked with -100
    p_len = formatted["prompt_tokens_count"]
    assert all(lbl == -100 for lbl in labels[:p_len])
    # The completion portion must have valid token IDs != -100
    assert any(lbl != -100 for lbl in labels[p_len:])


def test_agent_trajectories_schema():
    traj_file = PROJECT_ROOT / "datasets" / "agent_trajectories" / "agent_trajectories_train.jsonl"
    assert traj_file.exists()

    with open(traj_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 3
    for traj in lines:
        assert "task_id" in traj
        assert "user_request" in traj
        assert "steps" in traj
        assert len(traj["steps"]) >= 2
        for step in traj["steps"]:
            assert "tool_call" in step
            assert "observation" in step
            assert "name" in step["tool_call"]
            assert "arguments" in step["tool_call"]


def test_model_export_and_card():
    card_path = PROJECT_ROOT / "MODEL_CARD_27B.md"
    export_rep = PROJECT_ROOT / "reports" / "model_export_report.json"

    assert card_path.exists()
    assert export_rep.exists()

    card_text = card_path.read_text(encoding="utf-8")
    assert "27B" in card_text
    assert "LoRA SFT" in card_text
    assert "SHA-256" in card_text
