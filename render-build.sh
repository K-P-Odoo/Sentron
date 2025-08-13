#!/usr/bin/env bash
# Install system dependencies for dlib/face_recognition
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev

# Install Python packages
pip install -r requirements.txt
