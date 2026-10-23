FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY configs/ configs/
COPY packages/ packages/
COPY ml/ ml/
COPY agent/ agent/
COPY apps/ apps/
COPY reports/ reports/
COPY datasets/ datasets/
COPY tests/ tests/

RUN pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings sqlalchemy pytest httpx PyYAML torch numpy

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
