#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${1:?usage: run_pytest_app.sh <app-dir> [pytest args...]}"
shift || true

APP_PATH="$ROOT_DIR/$APP_DIR"

if [[ ! -d "$APP_PATH" ]]; then
  echo "app directory not found: $APP_DIR" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("$PYTHON_BIN")
elif [[ -x "$APP_PATH/.venv/bin/python" ]]; then
  python_cmd=("$APP_PATH/.venv/bin/python")
elif [[ -x "$APP_PATH/.venv/Scripts/python.exe" ]]; then
  python_cmd=("$APP_PATH/.venv/Scripts/python.exe")
elif command -v python3 >/dev/null 2>&1; then
  python_cmd=(python3)
elif command -v python >/dev/null 2>&1; then
  python_cmd=(python)
else
  echo "python is not available; install it or set PYTHON_BIN" >&2
  exit 1
fi

cd "$APP_PATH"
exec "${python_cmd[@]}" -m pytest "$@"
