# AI-Based Behavioral IDS - Docker Image
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package
RUN pip install -e .

# Create output directories
RUN mkdir -p outputs/models outputs/logs outputs/plots datasets

# Expose API port
EXPOSE 8080

# Default: run training
CMD ["python", "scripts/train_model.py", "--dataset", "synthetic", "--epochs", "10"]
