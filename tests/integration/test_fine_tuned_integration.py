"""
Integration and Regression Tests for Fine-Tuned Model Integration,
Rollback Support, and Continuous Improvement Loop.
Ensures Section 9, 10, and 11 deliverables remain fully operational.
"""

import sys
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry.registry import model_registry
from ml.inference.engine import inference_engine
from agent.feedback.continuous_improvement import improvement_loop, FailureCategory, Sanitizer


def test_fine_tuned_model_registered_and_loadable():
    # 1. Registry verification
    entry = model_registry.get_model("codeforge-h-sft-v1")
    assert entry is not None
    assert entry.model_id == "codeforge-h-sft-v1"
    assert "LoRA" in entry.architecture
    assert entry.estimated_vram_gb <= 4.0  # Must be within 4GB VRAM ceiling

    # 2. Feasibility check
    feasibility = model_registry.validate_loading_feasibility("codeforge-h-sft-v1")
    assert feasibility["feasible"] is True

    # 3. Load model
    res = model_registry.load_model("codeforge-h-sft-v1")
    assert res["success"] is True
    assert model_registry.active_model_id == "codeforge-h-sft-v1"


def test_rollback_mechanism():
    # Start at mock
    model_registry.load_model("codeforge-mock")
    assert model_registry.active_model_id == "codeforge-mock"

    # Switch to sft-v1
    model_registry.load_model("codeforge-h-sft-v1")
    assert model_registry.active_model_id == "codeforge-h-sft-v1"

    # Execute rollback
    rollback_res = model_registry.rollback_model()
    assert rollback_res["success"] is True
    assert rollback_res["rolled_back_from"] == "codeforge-h-sft-v1"
    assert model_registry.active_model_id == "codeforge-mock"

    # Inference engine rollback test
    model_registry.load_model("codeforge-h-sft-v1")
    ie_rollback = inference_engine.rollback()
    assert ie_rollback["success"] is True
    assert model_registry.active_model_id == "codeforge-mock"


def test_fine_tuned_inference():
    # Test generation with fine-tuned LoRA model
    out = inference_engine.generate(
        prompt="Write a Python health check endpoint",
        model_id="codeforge-h-sft-v1",
        backend="lora",
        thinking=True
    )
    assert "text" in out
    assert len(out["text"]) > 20
    assert out["reasoning"] is not None
    assert "LoRA SFT" in out["reasoning"]
    assert out["total_tokens"] > 0
    assert out["latency_ms"] >= 0.0


def test_sanitizer_removes_sensitive_paths_and_tokens():
    raw_trace = "FileNotFoundError in C:\\Users\\priya\\Desktop\\secret.py with api_key=sk-1234567890abcdef"
    clean = Sanitizer.sanitize_error(raw_trace)
    assert "C:\\Users\\<REDACTED>" in clean
    assert "api_key: <REDACTED_SECRET>" in clean
    assert "priya" not in clean


def test_continuous_improvement_pipeline():
    res = improvement_loop.record_failure(
        task_id="test_proj_regression",
        user_prompt="Build a user authentication microservice",
        failed_code="def broken(:\n    pass\n",
        raw_error="SyntaxError: invalid syntax in C:\\Users\\priya\\main.py",
        language="Python"
    )
    assert res["category"] == FailureCategory.MODEL_SYNTAX_ERROR.value
    assert "fingerprint" in res
    assert res["staged_samples_count"] >= 1

    # Gate check suppresses premature retraining
    gate = improvement_loop.check_and_trigger_retraining(force=False)
    assert "triggered" in gate
    assert "staged_count" in gate


def test_evaluation_reports_exist_and_verified():
    base_rep = PROJECT_ROOT / "reports" / "base_model_results.json"
    ft_rep = PROJECT_ROOT / "reports" / "fine_tuned_model_results.json"
    comp_md = PROJECT_ROOT / "reports" / "model_comparison.md"

    assert base_rep.exists()
    assert ft_rep.exists()
    assert comp_md.exists()

    with open(base_rep, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(ft_rep, "r", encoding="utf-8") as f:
        ft_data = json.load(f)

    # Validation loss improved
    assert ft_data["validation_loss"] < base_data["validation_loss"]
    # Pass rate improved
    assert ft_data["pass_at_1"] > base_data["pass_at_1"]
