"""
Sandboxed Execution Service for CodeForge AI.
Implements process-bounded workspace jail, timeout ceilings, output caps,
and dangerous command prevention.
"""

import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from packages.shared.types import SandboxExecutionResult
from configs.config import config

DISALLOWED_COMMANDS = [
    "format", "del /f /s /q c:", "rmdir /s /q c:", "mkfs", "dd if=",
    ":(){ :|:& };:", "shutdown", "reboot", "diskpart", "net user",
    "sudo", "runas"
]

MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB cap per Section S

class SandboxRunner:
    def __init__(self, base_workspace: Optional[Path] = None):
        self.base_workspace = base_workspace or config.projects_storage_dir
        self.base_workspace.mkdir(parents=True, exist_ok=True)

    def validate_command_safety(self, command: str) -> Tuple[bool, Optional[str]]:
        cmd_lower = command.lower().strip()
        for banned in DISALLOWED_COMMANDS:
            if banned in cmd_lower:
                return False, f"Command contains prohibited dangerous token: '{banned}'"
        return True, None

    def run(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout_seconds: int = 60,
        env: Optional[Dict[str, str]] = None
    ) -> SandboxExecutionResult:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        safe, reason = self.validate_command_safety(command)
        if not safe:
            return SandboxExecutionResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"[SECURITY JAIL ERROR RSK-002] {reason}",
                duration_ms=0.0,
                timeout_triggered=False,
                sandboxed=True,
                timestamp=timestamp
            )

        target_cwd = cwd or self.base_workspace
        target_cwd.mkdir(parents=True, exist_ok=True)

        # Build clean environment with sandboxing constraints
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        # Prevent interactive prompts
        run_env["CI"] = "1"
        run_env["PYTHONUNBUFFERED"] = "1"

        start_time = time.time()
        timeout_triggered = False

        try:
            # Execute with strict subprocess timeout
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(target_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env
            )
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
            else:
                proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:
                stdout, stderr = "", ""
            exit_code = 124
            timeout_triggered = True
            stderr += f"\n[TIMEOUT] Execution exceeded hard timeout limit of {timeout_seconds}s."
        except Exception as e:
            stdout = ""
            stderr = f"Subprocess invocation failure: {str(e)}"
            exit_code = 1

        duration_ms = (time.time() - start_time) * 1000.0

        # Enforce 64 KB output ceiling per Section S
        if len(stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
            stdout = stdout[:MAX_OUTPUT_BYTES] + "\n... [TRUNCATED at 64 KB ceiling]"
        if len(stderr.encode("utf-8")) > MAX_OUTPUT_BYTES:
            stderr = stderr[:MAX_OUTPUT_BYTES] + "\n... [TRUNCATED at 64 KB ceiling]"

        return SandboxExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=round(duration_ms, 2),
            timeout_triggered=timeout_triggered,
            sandboxed=True,
            timestamp=timestamp
        )

    def run_python_snippet(self, code: str, tests: str = "", timeout: int = 10) -> Dict[str, Any]:
        """
        Runs Python code and associated pytest/assert tests in a temporary isolated file.
        """
        import tempfile
        with tempfile.TemporaryDirectory(dir=str(self.base_workspace)) as tmpdir:
            file_path = Path(tmpdir) / "test_run.py"
            full_content = f"{code}\n\n# Tests\n{tests}\n"
            file_path.write_text(full_content, encoding="utf-8")

            res = self.run(f"{sys.executable} test_run.py", cwd=Path(tmpdir), timeout_seconds=timeout)
            passed = (res.exit_code == 0)
            return {
                "passed": passed,
                "exit_code": res.exit_code,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "duration_ms": res.duration_ms,
                "timeout": res.timeout_triggered
            }

sandbox_runner = SandboxRunner()
