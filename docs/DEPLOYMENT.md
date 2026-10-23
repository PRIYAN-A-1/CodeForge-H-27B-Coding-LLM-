# Deployment Guide (DEPLOYMENT.md)

## 1. Local Development Mode
- **Backend:** FastAPI running locally on `http://127.0.0.1:8000`.
- **Database:** SQLite database by default (`codeforge.db`).
- **Inference:** Mock backend (instant zero-latency CI) or local Ollama (`qwen3.8:27b`).
- **Web UI:** Interactive dashboard served directly by FastAPI or Next.js development server on `http://localhost:3000`.

## 2. Production Containerized Deployment
Docker Compose is provided (`docker-compose.yml`) for multi-container orchestration:
```bash
docker-compose up --build -d
```
Services:
1. `api`: FastAPI backend container with gunicorn/uvicorn workers.
2. `web`: Next.js production frontend.
3. `postgres`: PostgreSQL 16 database for production state persistence.
4. `sandbox`: Isolated container running untrusted test runners with `--network none`.
