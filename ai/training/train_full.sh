#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  echo "Create it with python3.12 -m venv .venv and install the CUDA/Python dependencies." >&2
  exit 2
fi
exec "$PYTHON" -m ai.training.full_train --directory ai/generated/full "$@"
