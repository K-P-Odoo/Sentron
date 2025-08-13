#!/usr/bin/env bash
set -Eeuo pipefail

echo ">>> Using Python: $(python -V)"

# Faster, non-interactive apt installs
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  libstdc++6 \
  libgl1 \
  libglib2.0-0 \
  python3-dev
# (If you use opencv-python-headless you can drop libgl1/libgtk-3-dev.)
rm -rf /var/lib/apt/lists/*

# Modern pip/build tooling
python -m pip install --upgrade pip setuptools wheel

# Prefer local wheels if present (no C++ compile on Render!)
if [ -d "wheels" ] && ls wheels/*.whl >/dev/null 2>&1; then
  echo ">>> Found local wheels/. Installing from wheels..."
  pip install --no-index --find-links=./wheels -r requirements.txt
else
  echo ">>> No local wheels found. Installing from PyPI..."
  pip install -r requirements.txt
fi

echo ">>> Build step complete."
