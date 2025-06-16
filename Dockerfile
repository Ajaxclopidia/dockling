# Stage 1: Build dependencies with CUDA tools
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies + Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        libopenblas-dev \
        ca-certificates \
        software-properties-common \
        curl \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm -rf /var/lib/apt/lists/* && \
    rm get-pip.py

# Setup virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy requirements file
COPY requirements.txt /build/requirements.txt

# Install PyTorch with CUDA 11.8 support and other dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir --upgrade \
        transformers \
        accelerate \
        docling-core \
        docling \
        fastapi \
        uvicorn[standard] \
        python-multipart && \
    pip install --no-cache-dir -r /build/requirements.txt

# Stage 2: Final runtime image  
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        ca-certificates \
        software-properties-common \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Copy application files
COPY main.py ./main.py
COPY test_script.py ./test_script.py 
COPY cli_client.py ./cli_client.py

# Create directories for uploads and temp files
RUN mkdir -p /app/uploads /app/temp

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]