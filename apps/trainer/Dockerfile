FROM python:3.11-slim

WORKDIR /app

COPY apps/trainer/pyproject.toml /app/apps/trainer/pyproject.toml
COPY apps/trainer/src /app/apps/trainer/src
COPY packages/pixel_sheriff_ml /app/packages/pixel_sheriff_ml

RUN pip install --no-cache-dir /app/packages/pixel_sheriff_ml \
  && pip install --no-cache-dir /app/apps/trainer

CMD ["python", "-m", "pixel_sheriff_trainer.main"]
