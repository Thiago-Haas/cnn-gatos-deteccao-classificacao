#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Erro: ambiente .venv não encontrado em ${PROJECT_DIR}." >&2
    echo "Prepare o ambiente executando: bash \"${PROJECT_DIR}/setup_venv.sh\"" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
exec "${VENV_DIR}/bin/python" "${PROJECT_DIR}/webcam.py" "$@"
