#!/usr/bin/env bash
set -Eeuo pipefail

echo ">>> Python: $(python -V)"
python -m pip install --upgrade pip setuptools wheel

if [ -d "wheels" ] && ls wheels/*.whl >/dev/null 2>&1; then
  echo ">>> Installing from local wheels/"
  pip install --no-index --find-links=./wheels -r requirements.txt
else
  echo ">>> Installing from PyPI"
  pip install -r requirements.txt
fi

echo ">>> Build complete."
