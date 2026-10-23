#!/usr/bin/env python3
"""
CodeForge AI - Unified Command Line Interface (CLI).
Provides developer commands for autonomous project planning, code generation,
sandbox testing, self-repair, foundation pretraining, LoRA fine-tuning,
evaluation benchmark harnesses, and local serving.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config
from packages.shared.types import ProjectRequirementInput, RepairRequest
from agent.planner.planner import requirement_planner
from agent.generator.generator import code_generator
from agent.repair.repair import repair_engine
from serve.pipeline import pipeline
from ml.model_registry.registry import model_registry


def cmd_plan(args):
    """Plan a project from natural language requirements."""
    req_input = ProjectRequirementInput(requirement=args.requirement)
    plan = requirement_planner.plan(req_input)
    if args.json:
        print(plan.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"  CODEFORGE AI PROJECT PLAN: {plan.project_name.upper()}")
        print("=" * 60)
        print(f"Project ID:      {plan.project_id}")
        print(f"Frontend:        {plan.frontend}")
        print(f"Backend:         {plan.backend}")
        print(f"Database:        {plan.database}")
        print("\nFeatures:")
        for feat in plan.features:
            print(f"  * {feat}")
        print("\nAcceptance Criteria:")
        for ac in plan.acceptance_criteria:
            print(f"  [x] {ac}")
        if plan.clarification_questions:
            print("\nClarifications:")
            for c in plan.clarification_questions:
                print(f"  ? {c}")
        print("=" * 60 + "\n")


def cmd_gen(args):
    """Generate a full autonomous project end-to-end."""
    print(f"\n[CodeForge] Executing autonomous generation for: '{args.requirement}'")
    result = pipeline.run(args.requirement)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print("\nExecution Trace:")
        for t in result.thinking_trace:
            print(f"  -> {t}")
        print("\n" + "-" * 60)
        print(f"Project ID:      {result.project_id}")
        print(f"Status:          {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Files Created:   {len(result.generation.files)}")
        print(f"Tests Passed:    {result.validation.tests_passed}")
        print(f"Tests Failed:    {result.validation.tests_failed}")
        print(f"Export ZIP:      {result.zip_path}")
        print(f"Summary:         {result.summary}")
        print("-" * 60 + "\n")


def cmd_test(args):
    """Run sandbox validation on a project."""
    project_id = args.project_id
    print(f"[CodeForge] Validating project '{project_id}' in isolated sandbox...")
    val = repair_engine.validate_project(project_id)
    if args.json:
        print(val.model_dump_json(indent=2))
    else:
        print(f"Validation Result: {'PASSED' if val.all_passed else 'FAILED'}")
        print(f"Tests Passed: {val.tests_passed}, Failed: {val.tests_failed}")
        print(f"Summary: {val.summary}")
        if val.unresolved_errors:
            print("Errors:")
            for err in val.unresolved_errors:
                print(f"  ! {err}")


def cmd_repair(args):
    """Run bounded self-repair on a project."""
    project_id = args.project_id
    val = repair_engine.validate_project(project_id)
    if val.all_passed:
        print(f"[CodeForge] Project '{project_id}' already has 100% passing checks. No repair needed.")
        return

    print(f"[CodeForge] Initiating self-repair on '{project_id}'...")
    req = RepairRequest(
        project_id=project_id,
        errors=val.unresolved_errors,
        max_attempts=args.max_attempts
    )
    outcome = repair_engine.repair_project(req)
    if args.json:
        print(outcome.model_dump_json(indent=2))
    else:
        print(f"Repair Status: {'SUCCESS' if outcome.success else 'FAILED'}")
        print(f"Attempts Used: {outcome.attempts_used} / {outcome.max_attempts}")
        for rec in outcome.history:
            print(f"  Attempt {rec.attempt_number}: {rec.files_modified} -> Post-test passed: {rec.test_result_after}")


def cmd_models(args):
    """List registered models."""
    models = model_registry.list_models()
    if args.json:
        print(json.dumps([m.model_dump() for m in models], indent=2))
    else:
        print("\n" + "=" * 70)
        print("  CODEFORGE AI MODEL REGISTRY")
        print("=" * 70)
        for m in models:
            active_marker = " [ACTIVE]" if m.model_id == model_registry.active_model_id else ""
            print(f"- {m.model_id}{active_marker}")
            print(f"  Architecture: {m.architecture} | Params: {m.parameter_count} | Context: {m.context_length}")
            print(f"  Backend: {m.backend} | VRAM: ~{m.estimated_vram_gb} GB | Precision: {m.precision}")
            print(f"  Description: {m.description}")
        print("=" * 70 + "\n")


def cmd_pretrain(args):
    """Run foundation pretraining."""
    from ml.training.pretraining.train import pretrain_smoke_test
    print(f"[CodeForge] Starting foundation pretraining (steps={args.steps})...")
    pretrain_smoke_test(steps=args.steps)


def cmd_sft(args):
    """Run Supervised Fine-Tuning (SFT / LoRA)."""
    from ml.training.finetuning.sft import sft_smoke_test
    print(f"[CodeForge] Starting Supervised Fine-Tuning (steps={args.steps})...")
    sft_smoke_test(steps=args.steps)


def cmd_eval(args):
    """Run evaluation benchmark harness."""
    from ml.evaluation.eval_harness import run_benchmark
    print("[CodeForge] Running evaluation benchmark harness...")
    res = run_benchmark()
    print(json.dumps(res, indent=2))


def cmd_hardware(args):
    """Display discovered hardware specification."""
    hw_file = PROJECT_ROOT / "reports" / "hardware_report.json"
    if hw_file.exists():
        with open(hw_file, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Hardware report not found. Run python scripts/discover_hardware.py first.")


def cmd_auto_train(args):
    """Run fully autonomous 27B training pipeline across all phases."""
    from scripts.run_autonomous_pipeline import AutonomousTrainingPipeline
    print(f"\n[CodeForge] Launching Autonomous 27B Training Pipeline (steps={args.steps}, budget={args.budget_gb}GB)...")
    runner = AutonomousTrainingPipeline(
        target_model=args.target_model,
        budget_gb=args.budget_gb,
        training_steps=args.steps,
        resume_checkpoint=args.resume,
        auto_mode=True
    )
    report = runner.run_all()
    if args.json:
        print(json.dumps(report, indent=2))


def cmd_serve(args):
    """Start the CodeForge AI FastAPI API server & Web Dashboard."""
    import uvicorn
    print(f"[CodeForge] Launching server on http://{args.host}:{args.port}")
    uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        prog="codeforge",
        description="CodeForge AI - Autonomous Multilingual Coding Model & Agent CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # auto-train
    p_auto = subparsers.add_parser("auto-train", help="Run fully autonomous 27B coding LLM training pipeline")
    p_auto.add_argument("--target-model", type=str, default="27b", help="Target model size")
    p_auto.add_argument("--steps", type=int, default=5, help="Training verification steps")
    p_auto.add_argument("--budget-gb", type=float, default=20.0, help="Storage budget in GB")
    p_auto.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    p_auto.add_argument("--json", action="store_true", help="Output final report as JSON")
    p_auto.set_defaults(func=cmd_auto_train)

    # plan
    p_plan = subparsers.add_parser("plan", help="Analyze requirements and build ProjectPlan")
    p_plan.add_argument("requirement", type=str, help="Natural language requirement prompt")
    p_plan.add_argument("--json", action="store_true", help="Output plan as JSON")
    p_plan.set_defaults(func=cmd_plan)

    # gen / run
    p_gen = subparsers.add_parser("gen", help="Autonomously generate, test, and package a project")
    p_gen.add_argument("requirement", type=str, help="Natural language requirement prompt")
    p_gen.add_argument("--json", action="store_true", help="Output result as JSON")
    p_gen.set_defaults(func=cmd_gen)

    # test
    p_test = subparsers.add_parser("test", help="Run sandbox verification on an existing project")
    p_test.add_argument("project_id", type=str, help="Project ID to test")
    p_test.add_argument("--json", action="store_true", help="Output validation as JSON")
    p_test.set_defaults(func=cmd_test)

    # repair
    p_rep = subparsers.add_parser("repair", help="Run bounded self-repair on a project")
    p_rep.add_argument("project_id", type=str, help="Project ID to repair")
    p_rep.add_argument("--max-attempts", type=int, default=3, help="Max repair retry attempts")
    p_rep.add_argument("--json", action="store_true", help="Output repair outcome as JSON")
    p_rep.set_defaults(func=cmd_repair)

    # models
    p_mod = subparsers.add_parser("models", help="List available model registry entries")
    p_mod.add_argument("--json", action="store_true", help="Output as JSON")
    p_mod.set_defaults(func=cmd_models)

    # pretrain
    p_pre = subparsers.add_parser("pretrain", help="Run model pretraining")
    p_pre.add_argument("--steps", type=int, default=10, help="Number of training steps")
    p_pre.set_defaults(func=cmd_pretrain)

    # sft
    p_sft = subparsers.add_parser("sft", help="Run LoRA fine-tuning")
    p_sft.add_argument("--steps", type=int, default=10, help="Number of SFT steps")
    p_sft.set_defaults(func=cmd_sft)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run pass@1 benchmark evaluation")
    p_eval.set_defaults(func=cmd_eval)

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Show hardware inspection report")
    p_hw.set_defaults(func=cmd_hardware)

    # serve
    p_srv = subparsers.add_parser("serve", help="Start the FastAPI API & web dashboard")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    p_srv.add_argument("--port", type=int, default=8000, help="Port")
    p_srv.add_argument("--reload", action="store_true", help="Enable hot reload")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
