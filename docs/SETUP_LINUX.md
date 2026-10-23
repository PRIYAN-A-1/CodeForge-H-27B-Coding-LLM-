# Linux & WSL2 Setup Guide (SETUP_LINUX.md)

Validated for Ubuntu 22.04 / 24.04 LTS and WSL2.

## Prerequisites
- Linux x86_64 or WSL2
- Python 3.11+
- Node.js v20+
- NVIDIA Drivers + CUDA Toolkit (Optional)

## 1. System Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git curl build-essential
```

## 2. Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Launch Backend
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Launch Frontend
```bash
cd apps/web
npm install
npm run dev
```

## 5. Run Verification Tests
```bash
pytest -v tests/
```
