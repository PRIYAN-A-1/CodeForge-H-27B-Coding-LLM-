"""
Unit tests for CodeForge AI Autonomous Serving Pipeline and CLI.
"""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serve.pipeline import AutonomousPipeline, PipelineResult
from cli.codeforge import cmd_plan, cmd_models, cmd_hardware
from packages.shared.types import ProjectRequirementInput
from agent.planner.planner import requirement_planner


def test_autonomous_pipeline_execution():
    pipeline = AutonomousPipeline()
    result = pipeline.run("Build a lightweight task tracking app with FastAPI and SQLite")

    assert isinstance(result, PipelineResult)
    assert result.success is True
    assert result.project_id.startswith("proj-")
    assert len(result.thinking_trace) >= 6
    assert len(result.generation.files) >= 5
    assert result.validation.all_passed is True
    assert result.validation.tests_passed > 0
    assert result.zip_path is not None
    assert Path(result.zip_path).exists()


def test_cli_plan_invocation(capsys):
    class MockArgs:
        requirement = "Build an invoice generator with Next.js and FastAPI"
        json = False

    cmd_plan(MockArgs())
    captured = capsys.readouterr()
    assert "CODEFORGE AI PROJECT PLAN" in captured.out
    assert "Next.js" in captured.out
    assert "FastAPI" in captured.out


def test_cli_models_invocation(capsys):
    class MockArgs:
        json = False

    cmd_models(MockArgs())
    captured = capsys.readouterr()
    assert "CODEFORGE AI MODEL REGISTRY" in captured.out
    assert "codeforge-h-pilot" in captured.out
    assert "codeforge-mock" in captured.out


def test_cli_hardware_invocation(capsys):
    class MockArgs:
        pass

    cmd_hardware(MockArgs())
    captured = capsys.readouterr()
    assert "AMD Ryzen" in captured.out or "hardware" in captured.out.lower()
