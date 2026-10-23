# Troubleshooting & Recovery Playbook (TROUBLESHOOTING.md)

Refer to Section X for common exceptions and mitigation steps:

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **`npm warn tar TAR_ENTRY_ERROR EBADF / EPERM` on G:** | Google Drive virtual volume file handle locking prevents deep node_modules extraction on Windows. | Run Next.js builds on a local physical NTFS drive (e.g. C:), or use the built-in FastAPI dashboard served directly at `http://127.0.0.1:8000/`. |
| **Ollama Request Timeout on 27B** | Cold-loading a 17 GB Q4 model into 4.4 GB free RAM requires heavy paging. | Allow 60–90s for first load, or switch to lightweight 0.5B/1.5B coding models or the built-in deterministic mock engine. |
| **`CUDA out of memory` during training** | Attempting large batch size or unquantized model on 6GB VRAM. | Run `python scripts/estimate_cost.py --tier pilot` before training. Use micro-batch size 1 with gradient accumulation, or switch to CPU smoke mode. |
| **Subprocess Timeout on Windows** | Runaway infinite loop in generated code. | `agent/execution/sandbox.py` automatically executes `taskkill /F /T /PID` to terminate the entire process tree. |
| **`ModuleNotFoundError: No module named 'packages'`** | Workspace root not in Python sys.path. | Run scripts from repository root, or set `$env:PYTHONPATH = "."` in PowerShell. |
