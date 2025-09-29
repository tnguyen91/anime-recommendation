FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY api/requirements.txt ./requirements.txt
RUN pip wheel --wheel-dir=/wheels --no-cache-dir -r requirements.txt


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-index --find-links=/wheels --no-cache-dir -r requirements.txt

COPY api ./api

EXPOSE 8000
ENV PORT=8000
ENTRYPOINT ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1} --timeout-keep-alive 60"]
