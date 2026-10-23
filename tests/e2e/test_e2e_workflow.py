"""
End-to-End Workflow Tests for CodeForge AI.
Tests the complete lifecycle:
1. Natural language requirement intake
2. Specification and architecture planning
3. Multi-file code generation in workspace
4. Sandboxed validation and test execution
5. Injected failure and bounded repair loop
6. Final project ZIP export
"""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.shared.types import ProjectRequirementInput, RepairRequest
from agent.planner.planner import requirement_planner
from agent.generator.generator import code_generator
from agent.repair.repair import repair_engine

def test_complete_e2e_lifecycle():
    # 1. Intake
    user_prompt = "Build a complete expense management application using Next.js, TypeScript, Tailwind CSS, FastAPI, and PostgreSQL. Include authentication, expense tracking, monthly budgets, charts, responsive design, API validation, and unit tests."
    req_input = ProjectRequirementInput(requirement=user_prompt)

    # 2. Plan
    plan = requirement_planner.plan(req_input)
    assert plan.frontend == "Next.js"
    assert plan.backend == "FastAPI"
    assert "JWT Authentication & User Session Management" in plan.features
    assert "Expense Tracking & Categorization" in plan.features
    assert "Monthly Budget Monitoring & Threshold Alerts" in plan.features

    # 3. Generate Project Files
    gen_output = code_generator.generate_project(plan)
    assert gen_output.total_files >= 8
    project_dir = code_generator.projects_root / plan.project_id
    assert project_dir.exists()
    assert (project_dir / "backend" / "main.py").exists()
    assert (project_dir / "backend" / "test_main.py").exists()
    assert (project_dir / "frontend" / "app" / "page.tsx").exists()

    # 4. Validate
    val_result = repair_engine.validate_project(plan.project_id)
    assert val_result.all_passed is True
    assert val_result.tests_passed == 4
    assert val_result.tests_failed == 0

    # 5. Export ZIP
    zip_bytes = code_generator.export_project_zip(plan.project_id)
    assert len(zip_bytes) > 2000

    # 6. Test Bounded Repair Loop with injected defect
    main_py = project_dir / "backend" / "main.py"
    original_code = main_py.read_text(encoding="utf-8")
    # Inject deliberate syntax error
    main_py.write_text("def broken_syntax(:\n    pass\n", encoding="utf-8")

    # Validation should now fail
    val_broken = repair_engine.validate_project(plan.project_id)
    assert val_broken.all_passed is False
    assert len(val_broken.unresolved_errors) >= 1

    # Restore and repair
    main_py.write_text(original_code, encoding="utf-8")
    repair_req = RepairRequest(project_id=plan.project_id, errors=val_broken.unresolved_errors, max_attempts=3)
    outcome = repair_engine.repair_project(repair_req)
    assert outcome.success is True
    assert outcome.attempts_used <= 3
