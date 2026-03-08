#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_USER="${DB_USER:-postgres}"
TEST_DB_NAME="${TEST_DB_NAME:-pixel_sheriff_test}"

cd "$ROOT_DIR"

docker compose up -d db redis >/dev/null

if ! docker compose exec -T db psql -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${TEST_DB_NAME}'" | grep -q 1; then
  docker compose exec -T db psql -U "$DB_USER" -d postgres -c "CREATE DATABASE ${TEST_DB_NAME}" >/dev/null
fi

if [[ "$#" -eq 0 ]]; then
  set -- python3 -m pytest -q
elif [[ "$1" == -* || "$1" == tests/* || "$1" == *.py ]]; then
  set -- python3 -m pytest "$@"
fi

# Rebuild the api-test image on invocation so test code and copied contract
# artifacts stay in sync with the current workspace.
docker compose --profile test run --build --rm api-test "$@"
