#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-assets}"
API_BASE="${DEMO_API_BASE_URL:-http://localhost:8010}"
WEB_BASE="${DEMO_WEB_BASE_URL:-http://localhost:3010}"

case "$MODE" in
  hero|screenshots|assets) ;;
  *)
    echo "Usage: ./scripts/run_demo_assets.sh [hero|screenshots|assets]" >&2
    exit 1
    ;;
esac

ensure_stack() {
  echo "Building Docker images for current web/api sources..."
  (
    cd "$ROOT_DIR"
    docker compose build api web-demo
    docker compose up -d db redis api web-demo
  )

  local attempts=60
  until curl -fsS "$API_BASE/api/v1/health" >/dev/null 2>&1 && curl -fsS "$WEB_BASE/projects" >/dev/null 2>&1; do
    attempts=$((attempts - 1))
    if [[ "$attempts" -le 0 ]]; then
      echo "Timed out waiting for web/api services to become ready." >&2
      exit 1
    fi
    sleep 2
  done
}

run_demo_assets() {
  (
    cd "$ROOT_DIR"
    docker compose run --rm demo-runner bash -lc "npm ci --cache /tmp/npm-cache && npm run demo:$MODE"
  )
}

ensure_stack
run_demo_assets
