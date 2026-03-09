# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY apps/api/pyproject.toml /app/apps/api/pyproject.toml
COPY apps/api/src /app/apps/api/src
COPY apps/worker/pyproject.toml /app/apps/worker/pyproject.toml
COPY apps/worker/src /app/apps/worker/src
COPY packages/pixel_sheriff_ml /app/packages/pixel_sheriff_ml
COPY packages/contracts /app/packages/contracts

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install /app/packages/pixel_sheriff_ml \
  && pip install /app/apps/api \
  && pip install /app/apps/worker

WORKDIR /app/apps/worker
CMD ["python", "-m", "sheriff_worker.main"]
