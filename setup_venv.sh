#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Erro: '${PYTHON_BIN}' não foi encontrado." >&2
    echo "Instale o Python 3 ou execute informando o interpretador:" >&2
    echo "  PYTHON_BIN=/caminho/para/python ./setup_venv.sh" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/requirements.txt" ]]; then
    echo "Erro: requirements.txt não encontrado em ${PROJECT_DIR}." >&2
    exit 1
fi

echo "Criando ambiente virtual em ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "Atualizando pip e ferramentas de instalação..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Instalando dependências do projeto..."
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo
echo "Ambiente pronto. Para ativá-lo, execute:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo
echo "Para abrir o notebook:"
echo "  jupyter lab"
