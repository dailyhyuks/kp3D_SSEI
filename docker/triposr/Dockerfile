# Wonder3D Docker Container for Korean Painting 3D Pipeline
# Multi-view diffusion based single-image-to-3D reconstruction

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglfw3 \
    libglfw3-dev \
    cmake \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install common dependencies
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    trimesh \
    transformers \
    accelerate \
    huggingface_hub \
    einops \
    omegaconf \
    pytorch-lightning \
    diffusers \
    safetensors \
    tqdm \
    opencv-python \
    scikit-image \
    scipy \
    matplotlib \
    rembg \
    gradio

# Install xformers for memory efficiency
RUN pip install --no-cache-dir xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# Clone Wonder3D
RUN git clone https://github.com/xxlong0/Wonder3D.git /app/Wonder3D

# Install Wonder3D dependencies
WORKDIR /app/Wonder3D
RUN pip install --no-cache-dir -r requirements.txt || true

# Install PyTorch3D (required for mesh processing)
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install NeuS for mesh extraction (Wonder3D uses this)
RUN pip install --no-cache-dir pymeshlab || true

# Copy inference script
COPY inference_wonder3d.py /app/inference_wonder3d.py

# Create directories for I/O and cache
RUN mkdir -p /input /output /cache

# Set working directory back to app
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["python", "/app/inference_wonder3d.py"]

# Default command
CMD ["--help"]
