#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

API_TEST_SERVICE="${API_TEST_SERVICE:-api-test-ml}" \
  bash "$ROOT_DIR/scripts/run_api_tests.sh" -q tests/ml "$@"
