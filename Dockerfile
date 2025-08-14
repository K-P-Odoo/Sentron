# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

# Runtime + build deps so dlib can compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-dev \
    libopenblas-dev liblapack-dev \
    libglib2.0-0 libgl1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Faster/cleaner builds
COPY requirements-cloudrun.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements-cloudrun.txt

# App code (incl. templates/static/models)
COPY . .

ENV PORT=8080
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 0 app_cloudrun:app
