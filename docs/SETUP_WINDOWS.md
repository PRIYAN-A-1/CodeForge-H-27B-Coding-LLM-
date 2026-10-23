# Windows Setup Guide (SETUP_WINDOWS.md)

This setup guide is validated for Windows 11 with PowerShell.

## Prerequisites
- Windows 11 64-bit
- Python 3.11+ (Python 3.13 supported)
- Node.js v20+ (Node v26 verified)
- Git for Windows
- NVIDIA GPU with driver 550+ (Optional, for GPU acceleration)
- Ollama (Optional, for local inference)

## 1. Clone & Environment Preparation
```powershell
# Open PowerShell in workspace root
cd "g:\My Drive\thoooo"

# Verify Python and Node
python --version
node --version
npm --version
```

## 2. Python Backend Setup
```powershell
# Install required packages (if not already present)
python -m pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy pytest pytest-asyncio httpx PyYAML torch

# Start FastAPI Backend Server
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```
The API is now live at `http://127.0.0.1:8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

## 3. Frontend Next.js Setup
```powershell
# In a separate terminal
cd apps\web
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

## 4. Run Automated Test Suite
```powershell
pytest -v tests/
```
All unit, integration, security, and end-to-end tests will execute in isolated sandboxes.
