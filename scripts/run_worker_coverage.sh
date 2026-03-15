#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/run_pytest_app.sh" apps/worker \
  --cov=sheriff_worker \
  --cov-report=term-missing \
  --cov-report=html:coverage/html \
  --cov-report=xml:coverage/coverage.xml \
  "$@"
