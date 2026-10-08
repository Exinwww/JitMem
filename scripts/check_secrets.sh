#!/usr/bin/env bash
# Scan staged content by default; never print credential values.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -x "${PROJECT_ROOT}/.tools/gitleaks" ]]; then
  SCANNER="${PROJECT_ROOT}/.tools/gitleaks"
elif command -v gitleaks >/dev/null 2>&1; then
  SCANNER="$(command -v gitleaks)"
else
  printf '%s\n' 'Secret scan requires Gitleaks. Install it with brew install gitleaks or place the official binary at .tools/gitleaks.' >&2
  exit 2
fi

COMMON=(--config="${PROJECT_ROOT}/.gitleaks.toml" --redact=100 --no-banner --no-color --ignore-gitleaks-allow)
case "${1:-staged}" in
  staged)
    exec "${SCANNER}" git . --pre-commit --staged "${COMMON[@]}"
    ;;
  history)
    exec "${SCANNER}" git . --log-opts='--all --full-history' "${COMMON[@]}"
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/check_secrets.sh [staged|history]' >&2
    exit 2
    ;;
esac
