#!/usr/bin/env bash
set -Eeuo pipefail

echo ">>> Python: $(python -V)"
python -m pip install --upgrade pip setuptools wheel

# Prefer local wheels, but allow PyPI for anything missing
pip install --find-links=./wheels -r requirements.txt

echo ">>> Build complete."
