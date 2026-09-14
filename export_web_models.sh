#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
fi
if ! "$PROJECT_DIR/.venv/bin/python" -c 'import torch, torchvision, ultralytics, onnx, onnxruntime, numpy' >/dev/null 2>&1; then
  "$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements-export.txt"
fi
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/export_web_models.py" "$@"
