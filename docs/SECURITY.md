# Security Architecture & Safeguards (SECURITY.md)

## 1. Principles
CodeForge AI treats all generated code, retrieved files, and benchmark samples as **untrusted input**.

## 2. Sandboxing & Process Boundaries
- **Working Directory Confinement:** Commands execute strictly within temporary project directories (`sandbox/projects/<project_id>`).
- **Path Traversal Guards:** All manifest file paths are verified against `..`, `/`, and `\` escaping. Any attempt to write outside the workspace raises a `ValueError` immediately.
- **Dangerous Command Blacklist:** Rejects system-altering commands (`format`, `del /f /s /q c:`, `shutdown`, `rmdir`, `net user`, `sudo`).
- **Timeout Ceilings:** Enforces a hard 10-15s timeout on child processes. On Windows, uses `taskkill /F /T /PID` to terminate the entire process tree and prevent orphaned runaway loops.
- **Output Capping:** Subprocess stdout and stderr are capped at 64 KB per Section S to prevent memory exhaustion via infinite print loops.

## 3. Secret & Credential Redaction
All ingested training data and user documents are scanned with regex patterns:
- Bearer tokens, private keys, and passwords -> `<KEY>`
- Email addresses -> `<EMAIL>`
- IP addresses -> `<IP>`
Database connection URLs with embedded passwords (`postgresql://user:pass@host`) are automatically redacted to `postgresql://user:<KEY>@host`.
