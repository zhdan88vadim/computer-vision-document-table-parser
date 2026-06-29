FROM python:3.10-slim

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements and install with PERMANENT cache
RUN --mount=type=cache,target=/root/.cache/pip,id=permanent_pip_cache,sharing=locked \
    pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY app.py .

# Command to run
CMD ["python", "app.py"]