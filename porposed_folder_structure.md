Monorepo folder structure (Next.js + FastAPI + Postgres + MAL-ready)
pixel-sheriff/
  README.md
  Roadplan.md
  Architecture.md
  docker-compose.yml
  .env.example
  .gitignore

  apps/
    web/                          # Next.js app (labeling UI)
      package.json
      next.config.js
      src/
        app/
        components/
        lib/
        styles/

    api/                          # FastAPI app (core CRUD + auth + export)
      pyproject.toml              # uv-managed deps
      uv.lock
      src/
        sheriff_api/
          __init__.py
          main.py                 # FastAPI entrypoint
          config.py
          db/
            session.py
            models.py
            migrations/           # Alembic
          routers/
            health.py
            projects.py
            categories.py
            assets.py
            annotations.py
            exports.py
          services/
            storage.py            # local/minio/s3 abstraction
            thumbnails.py
            video_frames.py       # frame extraction
            exporter_coco.py
            hashing.py
          schemas/
            projects.py
            categories.py
            assets.py
            annotations.py
            exports.py
          common/
            errors.py
            logging.py
      tests/

    worker/                       # Python worker (async jobs: frames, exports, MAL inference)
      pyproject.toml
      uv.lock
      src/
        sheriff_worker/
          __init__.py
          main.py                 # worker entrypoint
          jobs/
            extract_frames.py
            build_export_zip.py
            inference_suggest.py
          queues/
            broker.py             # redis connection + helpers
          common/
            logging.py
      tests/

  packages/
    contracts/                    # Shared “contracts” (generated TS client + schema docs)
      openapi/
      ts-client/                  # generated from FastAPI OpenAPI
      README.md

    ui/                           # Optional: shared UI components (later)
      package.json
      src/

  infra/
    db/
      init.sql
    nginx/
      dev.conf                    # optional reverse proxy


Why split api/ and worker/?

MAL and exports will want background execution (batch inference, frame extraction, zipping big datasets). Keeping it separate lets:

API stay responsive

jobs scale independently

you avoid “long request timeouts” and keep infra simple