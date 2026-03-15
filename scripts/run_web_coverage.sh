#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

node_path="$(command -v node || true)"
npm_path="$(command -v npm || true)"

if [[ -z "$node_path" || "$npm_path" == /mnt/c/Program\ Files/nodejs/* ]]; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    # Prefer the user's nvm-managed default Node instead of the Windows npm shim.
    # The suite is compatible with Node 20+ once it runs under a Linux node binary.
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
    nvm use default >/dev/null
  fi
fi

if ! command -v node >/dev/null 2>&1; then
  echo "node is not available; install it with nvm or add a Linux node binary to PATH" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not available; install it with nvm or add a Linux npm binary to PATH" >&2
  exit 1
fi

cd "$ROOT_DIR/apps/web"
npm run test:coverage -- "$@"
