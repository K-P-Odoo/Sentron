# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

# ---- System deps ----
# - build-essential, cmake, python3-dev: needed to compile dlib
# - openblas/lapack: faster linear algebra for dlib/np
# - libglib2.0-0 libgl1 libgomp1: runtime libs for OpenCV/TensorFlow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-dev \
    libopenblas-dev liblapack-dev \
    libglib2.0-0 libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip/setuptools/wheel once so builds are smoother
COPY requirements-cloudrun.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r requirements-cloudrun.txt

# Copy the whole app (incl. templates/static/models)
COPY . .

ENV PORT=8080
# IMPORTANT: start the Cloud Run entrypoint (your Cloud version), not app.py
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 0 wsgi:app
