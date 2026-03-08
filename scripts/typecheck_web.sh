#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

node_path="$(command -v node || true)"
npm_path="$(command -v npm || true)"

if [[ -z "$node_path" || "$npm_path" == /mnt/c/Program\ Files/nodejs/* ]]; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
    nvm use default >/dev/null
  fi
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is not available; install it with nvm or add a Linux npm binary to PATH" >&2
  exit 1
fi

cd "$ROOT_DIR/apps/web"
npx tsc --noEmit
