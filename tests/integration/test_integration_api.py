"""
Integration Tests for CodeForge AI FastAPI Endpoints:
- Health check
- Hardware status
- Model registry and load preflight
- Requirement planning
- Multi-file generation
- Validation endpoint
- Training preflight
- Token usage analytics
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_model" in data

def test_hardware_endpoint():
    res = client.get("/api/system/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "os" in data
    assert "cpu" in data
    assert "gpu" in data

def test_model_registry_and_load():
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 3

    # Loading valid mock model
    res_load = client.post("/api/models/load", json={"model_id": "codeforge-mock"})
    assert res_load.status_code == 200
    assert res_load.json()["success"] is True

    # Loading unquantized 27B model directly on 6GB VRAM must be rejected per Section 4/5
    res_reject = client.post("/api/models/load", json={"model_id": "codeforge-h-xl-27b"})
    assert res_reject.status_code == 400
    assert "exceeds available VRAM" in res_reject.json()["detail"]

def test_plan_and_generate_workflow():
    payload = {
        "requirement": "Build a complete expense management application with authentication, monthly budgets, and analytics."
    }
    # 1. Plan
    plan_res = client.post("/api/projects/plan", json=payload)
    assert plan_res.status_code == 200
    plan = plan_res.json()
    assert plan["frontend"] == "Next.js"
    assert plan["backend"] == "FastAPI"
    assert len(plan["features"]) >= 3

    # 2. Generate
    gen_res = client.post("/api/projects/generate", json=plan)
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["total_files"] >= 8
    assert len(gen_data["files"]) >= 8

    project_id = plan["project_id"]

    # 3. Validate
    val_res = client.post(f"/api/projects/{project_id}/validate")
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["all_passed"] is True
    assert val_data["tests_passed"] >= 3

    # 4. Export ZIP
    export_res = client.post(f"/api/projects/{project_id}/export")
    assert export_res.status_code == 200
    assert export_res.headers["content-type"] == "application/zip"
    assert len(export_res.content) > 1000  # Non-empty ZIP archive

def test_training_preflight():
    res = client.post("/api/training/preflight", json={"tier": "pilot"})
    assert res.status_code == 200
    data = res.json()
    assert data["passed_gate"] is True
    assert data["estimated_usd"] <= 300.0

def test_token_usage_endpoint():
    res = client.get("/api/usage/tokens")
    assert res.status_code == 200
    data = res.json()
    assert "total_tokens" in data
    assert "cumulative_cost_usd" in data
