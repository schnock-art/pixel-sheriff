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
  hero|screenshots|assets|prelabels) ;;
  *)
    echo "Usage: ./scripts/run_demo_assets.sh [hero|screenshots|assets|prelabels]" >&2
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

seed_demo_state() {
  local metadata_path="$ROOT_DIR/artifacts/demo/metadata/seed-demo-project.json"

  echo "Seeding deterministic demo project..."
  (
    cd "$ROOT_DIR"
    "$DOCKER_BIN" compose --profile demo run --rm demo-runner bash -lc "npm ci --cache /tmp/npm-cache && node ../../scripts/demo/seed-demo-project.mjs --force >/dev/null"
  )

  if [[ ! -f "$metadata_path" ]]; then
    echo "Demo seed metadata was not created at $metadata_path" >&2
    exit 1
  fi

  if [[ "$MODE" != "prelabels" ]]; then
    return
  fi

  local prelabel_metadata_path="$ROOT_DIR/artifacts/demo/metadata/prelabel-demo.json"
  local project_id
  local task_id
  project_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["projectId"])' "$metadata_path")"
  task_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["taskId"])' "$metadata_path")"

  echo "Injecting deterministic AI prelabel review data..."
  (
    cd "$ROOT_DIR"
    "$DOCKER_BIN" compose --profile demo exec -T api-demo python -m sheriff_api.demo_prelabel_seed "$project_id" "$task_id" > "$prelabel_metadata_path"
  )

  python3 - <<'PY' "$metadata_path" "$prelabel_metadata_path"
import json
import sys

metadata_path, prelabel_metadata_path = sys.argv[1], sys.argv[2]
with open(metadata_path, encoding="utf-8") as handle:
    metadata = json.load(handle)
with open(prelabel_metadata_path, encoding="utf-8") as handle:
    metadata["prelabelDemo"] = json.load(handle)
with open(metadata_path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY
}

ensure_stack
seed_demo_state
run_demo_assets
