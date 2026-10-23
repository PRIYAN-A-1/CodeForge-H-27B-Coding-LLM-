# CodeForge AI — VS Code Extension

Autonomous full-stack project generation, sandbox validation, and self-repair powered by CodeForge-H.

## Features

- **CodeForge: Generate Full-Stack Project from Requirement**: Turn any natural language description into an entire multi-file project (Next.js, FastAPI, SQLite/PostgreSQL, Tailwind CSS, Pytest, Dockerfile) in seconds.
- **CodeForge: Validate Current Project**: Execute isolated sandbox tests, syntax validation, and security scans.
- **CodeForge: Self-Repair Failing Project Tests**: Bounded iterative automated patching when compiler or test errors are detected.
- **CodeForge: Open Web Dashboard**: Quick one-click navigation to the rich interactive CodeForge AI web dashboard.

## Installation

1. Package the extension:
   ```bash
   cd ide/vscode_extension
   npm install -g @vscode/vsce
   vsce package
   ```
2. Install in VS Code:
   ```bash
   code --install-extension codeforge-ai-1.0.0.vsix
   ```

## Configuration

Set `codeforge.apiUrl` in VS Code settings to point to your running CodeForge AI backend (defaults to `http://127.0.0.1:8000`).
