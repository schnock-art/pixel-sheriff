#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_USER="${DB_USER:-postgres}"
TEST_DB_NAME="${TEST_DB_NAME:-pixel_sheriff_test}"

pytest_args=(
  --cov=sheriff_api
  --cov-report=term-missing
  --cov-report=html:coverage/html
  --cov-report=xml:coverage/coverage.xml
)

if command -v python3 >/dev/null 2>&1; then
  local_python=(python3)
elif command -v python >/dev/null 2>&1; then
  local_python=(python)
else
  local_python=()
fi

if [[ "$#" -gt 0 ]]; then
  pytest_args+=("$@")
else
  pytest_args+=(-q --ignore=tests/ml tests)
fi

run_local() {
  echo "Docker Compose unavailable; running API coverage locally from apps/api" >&2
  if [[ "${#local_python[@]}" -eq 0 ]]; then
    echo "python is not available; install it or run API coverage through Docker Compose" >&2
    exit 1
  fi
  cd "$ROOT_DIR/apps/api"
  exec "${local_python[@]}" -m pytest "${pytest_args[@]}"
}

cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  run_local
fi

if ! docker compose version >/dev/null 2>&1; then
  run_local
fi

docker compose --profile test up -d db-test redis-test >/dev/null

if ! docker compose --profile test exec -T db-test psql -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${TEST_DB_NAME}'" | grep -q 1; then
  docker compose --profile test exec -T db-test psql -U "$DB_USER" -d postgres -c "CREATE DATABASE ${TEST_DB_NAME}" >/dev/null
fi

docker compose --profile test run --build --rm api-test python3 -m pytest "${pytest_args[@]}"
