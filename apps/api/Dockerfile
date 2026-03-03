# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

COPY apps/api/pyproject.toml /app/apps/api/pyproject.toml
COPY apps/api/src /app/apps/api/src
COPY packages/pixel_sheriff_ml /app/packages/pixel_sheriff_ml

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install /app/packages/pixel_sheriff_ml \
  && pip install /app/apps/api

WORKDIR /app/apps/api
CMD ["uvicorn", "sheriff_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
