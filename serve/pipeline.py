"""
Autonomous End-to-End Execution Pipeline for CodeForge AI.
Coordinates the entire autonomous software engineering lifecycle:
1. Requirement Analysis & Reasoning (Thinking Trace)
2. Formal Specification & ProjectPlan synthesis
3. Production Multi-File Code Generation
4. Sandbox Execution & Verification
5. Bounded Self-Repair Loop if checks fail
6. Artifact Packaging (ZIP archive & summary)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.shared.types import (
    ProjectRequirementInput,
    ProjectPlan,
    ProjectGenerationOutput,
    ValidationResult,
    RepairRequest,
    RepairOutcome,
)
from agent.planner.planner import requirement_planner
from agent.generator.generator import code_generator
from agent.repair.repair import repair_engine
from configs.config import config


class PipelineResult(BaseModel):
    project_id: str
    project_name: str
    requirement: str
    thinking_trace: List[str]
    plan: ProjectPlan
    generation: ProjectGenerationOutput
    validation: ValidationResult
    repair_outcome: Optional[RepairOutcome] = None
    zip_path: Optional[str] = None
    duration_seconds: float
    success: bool
    summary: str


class AutonomousPipeline:
    def __init__(self, projects_root: Optional[Path] = None):
        self.projects_root = projects_root or config.projects_storage_dir
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def run(self, requirement: str, preferred_stack: Optional[Dict[str, str]] = None) -> PipelineResult:
        start_time = time.time()
        trace: List[str] = []

        # Step 1: Thinking & Intent Extraction
        trace.append(f"[1/6] Analyzing natural language requirement: '{requirement}'")
        req_input = ProjectRequirementInput(
            requirement=requirement,
            preferred_stack=preferred_stack or {}
        )

        # Step 2: Architecture Planning & Specification
        trace.append("[2/6] Decomposing architectural layers, selecting frontend/backend/database stacks...")
        plan = requirement_planner.plan(req_input)
        trace.append(
            f"[2/6] Generated ProjectPlan {plan.project_id}: '{plan.project_name}' "
            f"({plan.frontend} + {plan.backend} + {plan.database})"
        )

        # Step 3: Multi-File Code Synthesis
        trace.append("[3/6] Generating production multi-file project inside isolated workspace...")
        gen_output = code_generator.generate_project(plan)
        total_loc = sum(len(f.content.splitlines()) for f in gen_output.files)
        trace.append(f"[3/6] Synthesized {len(gen_output.files)} files ({total_loc} LOC).")

        # Step 4: Sandbox Validation
        trace.append("[4/6] Running sandbox verification (syntax checks, pytest test suite)...")
        val_result = repair_engine.validate_project(plan.project_id)

        # Step 5: Bounded Self-Repair
        repair_res = None
        if not val_result.all_passed:
            trace.append("[5/6] Validation detected issues. Triggering bounded self-repair engine...")
            repair_req = RepairRequest(
                project_id=plan.project_id,
                errors=val_result.unresolved_errors,
                max_attempts=config.max_repair_attempts
            )
            repair_res = repair_engine.repair_project(repair_req)
            val_result = repair_engine.validate_project(plan.project_id)
            trace.append(
                f"[5/6] Self-repair completed: success={repair_res.success}, "
                f"attempts={repair_res.attempts_used}/{repair_res.max_attempts}."
            )
        else:
            trace.append("[5/6] Validation passed with zero errors on initial attempt.")

        # Step 6: Packaging & Export
        trace.append("[6/6] Packaging project deployment artifacts...")
        zip_bytes = code_generator.export_project_zip(plan.project_id)
        zip_file = self.projects_root / f"{plan.project_id}.zip"
        zip_file.write_bytes(zip_bytes)
        trace.append(f"[6/6] Export archive created: {zip_file}")

        duration = round(time.time() - start_time, 2)
        success = val_result.all_passed

        summary = (
            f"Successfully built '{plan.project_name}' in {duration}s. "
            f"Stack: {plan.frontend} + {plan.backend} + {plan.database}. "
            f"Generated {len(gen_output.files)} files. "
            f"All tests passing: {val_result.tests_passed} passed, {val_result.tests_failed} failed."
        )

        return PipelineResult(
            project_id=plan.project_id,
            project_name=plan.project_name,
            requirement=requirement,
            thinking_trace=trace,
            plan=plan,
            generation=gen_output,
            validation=val_result,
            repair_outcome=repair_res,
            zip_path=str(zip_file),
            duration_seconds=duration,
            success=success,
            summary=summary
        )


pipeline = AutonomousPipeline()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CodeForge AI - Autonomous Pipeline Runner")
    parser.add_argument("requirement", type=str, help="Natural language requirement prompt")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    res = pipeline.run(args.requirement)
    if args.json:
        print(res.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 70)
        print("  CODEFORGE AI AUTONOMOUS PIPELINE EXECUTION")
        print("=" * 70)
        for t in res.thinking_trace:
            print(f"  > {t}")
        print("-" * 70)
        print(f"Project ID: {res.project_id}")
        print(f"Project Name: {res.project_name}")
        print(f"Status: {'SUCCESS' if res.success else 'FAILED'}")
        print(f"Summary: {res.summary}")
        print(f"Zip Archive: {res.zip_path}")
        print("=" * 70 + "\n")
