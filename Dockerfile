FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_NAME=app
ARG UID=1000
ARG GID=1000

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies, copy the workspace, then install Python deps
# Set working directory
WORKDIR /workspace

# Install OS packages commonly required for vision, audio and building Python packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		git \
		curl \
		ca-certificates \
		build-essential \
		pkg-config \
		ffmpeg \
		libgl1-mesa-glx \
		libsm6 \
		libxrender1 \
		libxext6 \
		ssh \
		rsync \
    && update-ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Copy repository into the image
COPY . /workspace

# Upgrade pip and install Python dependencies if present, then install the project as a source module
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e git+https://github.com/NIRVANALAN/STream3R#egg=stream3r

# Make sure python can import the local package
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command: run the demo (can be overridden at runtime)
# CMD ["python", "demo.py"]

