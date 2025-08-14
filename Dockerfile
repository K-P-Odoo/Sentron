# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

# Runtime libs needed by OpenCV/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install server deps (make sure requirements-cloudrun.txt pins dlib==20.0.0)
COPY requirements-cloudrun.txt .
RUN python -m pip install --upgrade pip && \
    pip install --prefer-binary -r requirements-cloudrun.txt

# Copy the whole app (incl. templates/static/models)
COPY . .

ENV PORT=8080

# IMPORTANT: start the Cloud Run entrypoint (not app.py)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 0 app_cloudrun:app
