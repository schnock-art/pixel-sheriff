#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-assets}"
API_BASE="${DEMO_API_BASE_URL:-http://localhost:8010}"
WEB_BASE="${DEMO_WEB_BASE_URL:-http://localhost:3010}"
DOCKER_BIN="$(command -v docker || true)"

if [[ -z "$DOCKER_BIN" && -x "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]]; then
  DOCKER_BIN="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
fi

if [[ -z "$DOCKER_BIN" ]]; then
  echo "docker is not available; install Docker Desktop with WSL integration or add docker to PATH" >&2
  exit 1
fi

case "$MODE" in
  hero|screenshots|assets) ;;
  *)
    echo "Usage: ./scripts/run_demo_assets.sh [hero|screenshots|assets]" >&2
    exit 1
    ;;
esac

ensure_stack() {
  echo "Building isolated demo images for current web/api sources..."
  (
    cd "$ROOT_DIR"
    "$DOCKER_BIN" compose --profile demo build api-demo trainer-demo web-demo
    "$DOCKER_BIN" compose --profile demo up -d db-demo redis-demo trainer-demo api-demo web-demo
  )

  local attempts=60
  until (
    cd "$ROOT_DIR" &&
    "$DOCKER_BIN" compose --profile demo exec -T api-demo python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" >/dev/null 2>&1 &&
    "$DOCKER_BIN" compose --profile demo exec -T web-demo sh -lc "wget -q -O - http://localhost:3000/projects >/dev/null"
  ); do
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
    "$DOCKER_BIN" compose --profile demo run --rm demo-runner bash -lc "npm ci --cache /tmp/npm-cache && npm run demo:$MODE"
  )
}

ensure_stack
run_demo_assets
