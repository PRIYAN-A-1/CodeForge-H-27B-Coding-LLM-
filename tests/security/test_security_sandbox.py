"""
Security & Sandbox Isolation Tests for CodeForge AI.
Verifies Section 17, 22, and 24:
- Path traversal attacks
- Prohibited and destructive command injection
- Timeout enforcement for runaway processes (e.g. infinite loops)
- Output size truncation at 64 KB ceiling
- Secret redaction and credential protection
"""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.execution.sandbox import sandbox_runner
from packages.shared.types import FileManifestItem
from ml.data.preprocessing.clean_dataset import clean_code_text

def test_command_injection_prohibition():
    destructive_cmds = [
        "format c:",
        "del /f /s /q c:\\",
        "rmdir /s /q c:\\",
        "shutdown /s /t 0",
        "net user evil /add"
    ]
    for cmd in destructive_cmds:
        res = sandbox_runner.run(cmd)
        assert res.exit_code == 126
        assert "SECURITY JAIL ERROR RSK-002" in res.stderr

def test_runaway_infinite_loop_timeout():
    # Runaway Python infinite loop
    loop_cmd = f'"{sys.executable}" -c "while True: pass"'
    res = sandbox_runner.run(loop_cmd, timeout_seconds=2)
    assert res.timeout_triggered is True
    assert res.exit_code == 124
    assert "[TIMEOUT]" in res.stderr

def test_output_buffer_cap_at_64kb():
    # Generates 150 KB of stdout
    big_output_cmd = f'"{sys.executable}" -c "import sys; sys.stdout.write(\'A\' * 150000)"'
    res = sandbox_runner.run(big_output_cmd, timeout_seconds=5)
    assert len(res.stdout.encode("utf-8")) <= 66000
    assert "[TRUNCATED at 64 KB ceiling]" in res.stdout

def test_path_traversal_manifest_rejection():
    attack_vectors = [
        "../../etc/shadow",
        "..\\..\\Windows\\System32\\cmd.exe",
        "/absolute/path/file.txt",
        "\\absolute\\path\\file.txt"
    ]
    for vec in attack_vectors:
        with pytest.raises(ValueError):
            FileManifestItem(
                path=vec,
                file_type="python",
                purpose="attack",
                content="malicious"
            )

def test_secret_redaction_guarantee():
    sensitive_text = (
        "AWS_SECRET_ACCESS_KEY = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'\n"
        "GITHUB_TOKEN = 'ghp_1234567890abcdefghijklmnopqrstuvwxyz12'\n"
        "DATABASE_URL = 'postgresql://admin:super_secret_password_123@192.168.1.50:5432/db'\n"
    )
    keep, clean, _ = clean_code_text(sensitive_text)
    assert keep
    assert "<KEY>" in clean
    assert "<IP>" in clean
    assert "super_secret_password_123" not in clean
    assert "ghp_" not in clean
