# syntax=docker/dockerfile:1.7
ARG TRAINER_BASE_IMAGE=python:3.11-slim
FROM ${TRAINER_BASE_IMAGE}

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

WORKDIR /app

# Fallback: if the selected base image does not already contain torch/torchvision,
# install them here.
RUN --mount=type=cache,target=/root/.cache/pip \
  python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision') else 1)" \
  || pip install --index-url ${PYTORCH_INDEX_URL} torch torchvision

COPY apps/trainer/pyproject.toml /app/apps/trainer/pyproject.toml
COPY apps/trainer/src /app/apps/trainer/src
COPY packages/pixel_sheriff_ml /app/packages/pixel_sheriff_ml

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install /app/packages/pixel_sheriff_ml \
  && pip install /app/apps/trainer

CMD ["python", "-m", "pixel_sheriff_trainer.main"]
