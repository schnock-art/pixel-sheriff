FROM python:3.11-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

WORKDIR /app

COPY apps/trainer/pyproject.toml /app/apps/trainer/pyproject.toml
COPY apps/trainer/src /app/apps/trainer/src
COPY packages/pixel_sheriff_ml /app/packages/pixel_sheriff_ml

RUN pip install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} torch torchvision \
  && pip install --no-cache-dir /app/packages/pixel_sheriff_ml \
  && pip install --no-cache-dir /app/apps/trainer

CMD ["python", "-m", "pixel_sheriff_trainer.main"]
