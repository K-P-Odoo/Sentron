# Dockerfile
FROM python:3.10.11-slim

# Helpful Python defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps needed by dlib/face_recognition (no GUI libs needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# App dir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code & model files
COPY . .

# Start with gunicorn; bind to $PORT (Render/Cloud Run set it), default to 8080 locally
CMD ["bash","-lc","gunicorn app:app -k gthread --threads 2 --workers 2 --timeout 120 --bind 0.0.0.0:${PORT:-8080}"]
