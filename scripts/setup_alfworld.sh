#!/usr/bin/env bash
# Install only the text environment and test tools; keep the existing data intact.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-/Users/linbei/workspace/experiential_memory/data/alfworld}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/jitmem-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${PROJECT_ROOT}/.python"
export UV_PYTHON_BIN_DIR="${PROJECT_ROOT}/.python/bin"
export ALFWORLD_DATA="${DATA_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'This setup needs uv. Install uv first, or create a Python 3.11 environment and run pip install -e ".[alfworld,dev]".' >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/json_2.1.1" ]]; then
  printf 'ALFWorld data root does not contain json_2.1.1: %s\n' "${DATA_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
# TextWorld 1.6.2 supplies native Apple Silicon wheels. Version 1.7.0 does not.
uv python install 3.11.16
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11.16 .venv
fi
uv pip install --python .venv/bin/python -c requirements-lock.txt -e '.[alfworld,dev]'
.venv/bin/python -c 'from importlib.metadata import version; print("ALFWorld", version("alfworld"), "TextWorld", version("textworld"))'
printf 'Environment ready: %s/.venv/bin/python\n' "${PROJECT_ROOT}"
printf 'Data root (read only during evaluation): %s\n' "${DATA_ROOT}"
