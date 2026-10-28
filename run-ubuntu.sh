#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export STROKESENSE_DEVICE="${STROKESENSE_DEVICE:-gpu}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
exec .venv-ubuntu/bin/python scripts/serve.py
