#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing macOS virtual environment. Follow RUN_MACOS.md first." >&2
  exit 1
fi
export STROKESENSE_DEVICE=cpu
exec .venv/bin/python scripts/serve.py
