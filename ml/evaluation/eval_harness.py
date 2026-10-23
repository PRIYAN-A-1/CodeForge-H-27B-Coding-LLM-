"""
Comprehensive Evaluation Harness for CodeForge-H.
Implements Section 19 & Section Q:
- Unbiased pass@k estimation: pass_at_k(n, c, k)
- Benchmark fixture verification: reference solutions score 100%, stub scores 0%
- Private Requirements Benchmark runner
- Markdown & JSON reporting
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. (HumanEval).
    Formula: 1.0 - prod(1 - k / i) for i in [n - c + 1, ..., n]
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

# Canonical Evaluation Benchmark Fixtures
EVAL_FIXTURES = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n",
        "reference": (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = abs(elem - elem2)\n"
            "                if distance < threshold:\n"
            "                    return True\n"
            "    return False\n"
        ),
        "test": (
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
        )
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n",
        "reference": (
            "def separate_paren_groups(paren_string: str) -> list[str]:\n"
            "    result = []\n"
            "    current_string = []\n"
            "    current_depth = 0\n"
            "    for c in paren_string:\n"
            "        if c == '(':\n"
            "            current_depth += 1\n"
            "            current_string.append(c)\n"
            "        elif c == ')':\n"
            "            current_depth -= 1\n"
            "            current_string.append(c)\n"
            "            if current_depth == 0:\n"
            "                result.append(''.join(current_string))\n"
            "                current_string.clear()\n"
            "    return result\n"
        ),
        "test": (
            "assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n"
            "assert separate_paren_groups('() (()) ((()))') == ['()', '(())', '((()))']\n"
        )
    },
    {
        "task_id": "Private/001",
        "prompt": "def calculate_budget_remaining(budget: float, expenses: list[float]) -> dict:\n",
        "reference": (
            "def calculate_budget_remaining(budget: float, expenses: list[float]) -> dict:\n"
            "    total = sum(expenses)\n"
            "    return {'total_spent': round(total, 2), 'remaining': round(budget - total, 2), 'over_budget': total > budget}\n"
        ),
        "test": (
            "res = calculate_budget_remaining(100.0, [25.0, 40.0])\n"
            "assert res['remaining'] == 35.0 and not res['over_budget']\n"
            "res2 = calculate_budget_remaining(50.0, [60.0])\n"
            "assert res2['remaining'] == -10.0 and res2['over_budget']\n"
        )
    }
]

class EvaluationHarness:
    def __init__(self):
        from agent.execution.sandbox import sandbox_runner
        self.runner = sandbox_runner

    def evaluate_candidate(self, code: str, tests: str) -> bool:
        res = self.runner.run_python_snippet(code, tests, timeout=5)
        return res["passed"]

    def run_benchmark(self, code_generator_fn: Callable[[str], str]) -> Dict[str, Any]:
        results = []
        for task in EVAL_FIXTURES:
            gen_code = code_generator_fn(task["prompt"])
            passed = self.evaluate_candidate(gen_code, task["test"])
            results.append({
                "task_id": task["task_id"],
                "passed": passed
            })

        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        pass_at_1 = pass_at_k(n=1, c=1 if passed_count == total else (passed_count // total), k=1)

        return {
            "total_tasks": total,
            "passed_tasks": passed_count,
            "pass_at_1": round(passed_count / total, 4),
            "results": results
        }

    def validate_harness_fixtures(self) -> Dict[str, Any]:
        """
        Validates Section Q harness invariant:
        Reference solutions score 100%, while a 'return None' stub scores 0%.
        """
        ref_results = [self.evaluate_candidate(t["reference"], t["test"]) for t in EVAL_FIXTURES]
        stub_results = [self.evaluate_candidate(t["prompt"] + "    return None\n", t["test"]) for t in EVAL_FIXTURES]

        ref_pass_rate = sum(1 for r in ref_results if r) / len(ref_results)
        stub_pass_rate = sum(1 for s in stub_results if s) / len(stub_results)

        assert ref_pass_rate == 1.0, f"Harness failure: Reference solutions scored {ref_pass_rate * 100}% instead of 100%"
        assert stub_pass_rate == 0.0, f"Harness failure: Stub solutions scored {stub_pass_rate * 100}% instead of 0%"

        return {
            "harness_verified": True,
            "reference_score": 1.0,
            "stub_score": 0.0,
            "num_fixtures": len(EVAL_FIXTURES)
        }

evaluation_harness = EvaluationHarness()

def run_benchmark(code_generator_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    if code_generator_fn is None:
        # Default to validating harness fixtures and reference baseline
        return evaluation_harness.validate_harness_fixtures()
    return evaluation_harness.run_benchmark(code_generator_fn)

if __name__ == "__main__":
    harness = evaluation_harness
    val = harness.validate_harness_fixtures()
    print(f"Harness Validation Result: {val}")

    # Pass@k formula unit check
    assert pass_at_k(20, 0, 1) == 0.0
    assert pass_at_k(20, 20, 1) == 1.0
    print("Pass@k formula verified successfully.")
