# Real-ESRGAN Docker Image for Image Preprocessing
# Use NVIDIA PyTorch NGC base image for CUDA 13.0 support
# Note: Using older PyTorch NGC image that supports Python 3.10 for Real-ESRGAN compatibility
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working directory
WORKDIR /app

# Install system dependencies required by Real-ESRGAN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables (CUDA 13.0 only supports sm_89, sm_90, sm_121)
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.1"
ENV FORCE_CUDA=1

# NGC container already has PyTorch with CUDA 13.0 support
# Install Real-ESRGAN dependencies directly with pip (no conda needed)
# This layer will be cached unless dependencies change
RUN pip install --no-cache-dir \
    "numpy<2" \
    opencv-python==4.8.0.74 \
    Pillow \
    "basicsr>=1.4.2" \
    "facexlib>=0.2.5" \
    "gfpgan>=1.3.5" \
    tqdm \
    addict \
    future \
    lmdb \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    tb-nightly \
    yapf \
    filterpy \
    numba \
    realesrgan

# Verify CUDA availability (NGC PyTorch already has CUDA support)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Copy application files (do this last - changes frequently)
COPY . .

# Create necessary directories
RUN mkdir -p /app/weights /app/inputs /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create entrypoint script for batch processing (no conda needed)
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -e' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'INPUT_DIR=${INPUT_DIR:-/app/inputs}' >> /app/entrypoint.sh && \
    echo 'OUTPUT_DIR=${OUTPUT_DIR:-/app/outputs}' >> /app/entrypoint.sh && \
    echo 'MODEL_NAME=${MODEL_NAME:-RealESRGAN_x4plus}' >> /app/entrypoint.sh && \
    echo 'OUTSCALE=${OUTSCALE:-2}' >> /app/entrypoint.sh && \
    echo 'TILE=${TILE:-400}' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'echo "==========================================="' >> /app/entrypoint.sh && \
    echo 'echo "Real-ESRGAN Image Enhancement"' >> /app/entrypoint.sh && \
    echo 'echo "==========================================="' >> /app/entrypoint.sh && \
    echo 'echo "Input directory: $INPUT_DIR"' >> /app/entrypoint.sh && \
    echo 'echo "Output directory: $OUTPUT_DIR"' >> /app/entrypoint.sh && \
    echo 'echo "Model: $MODEL_NAME"' >> /app/entrypoint.sh && \
    echo 'echo "Output scale: ${OUTSCALE}x"' >> /app/entrypoint.sh && \
    echo 'echo "Tile size: $TILE"' >> /app/entrypoint.sh && \
    echo 'echo "==========================================="' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'python /app/inference_realesrgan.py \' >> /app/entrypoint.sh && \
    echo '    -i "$INPUT_DIR" \' >> /app/entrypoint.sh && \
    echo '    -o "$OUTPUT_DIR" \' >> /app/entrypoint.sh && \
    echo '    -n "$MODEL_NAME" \' >> /app/entrypoint.sh && \
    echo '    -s "$OUTSCALE" \' >> /app/entrypoint.sh && \
    echo '    --tile "$TILE" \' >> /app/entrypoint.sh && \
    echo '    --tile_pad 10 \' >> /app/entrypoint.sh && \
    echo '    --pre_pad 0 \' >> /app/entrypoint.sh && \
    echo '    --fp32 \' >> /app/entrypoint.sh && \
    echo '    "$@"' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Fix file permissions for host access' >> /app/entrypoint.sh && \
    echo 'chmod -R 777 "$OUTPUT_DIR"' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'echo ""' >> /app/entrypoint.sh && \
    echo 'echo "==========================================="' >> /app/entrypoint.sh && \
    echo 'echo "Enhancement Complete!"' >> /app/entrypoint.sh && \
    echo 'echo "Output saved to: $OUTPUT_DIR"' >> /app/entrypoint.sh && \
    echo 'echo "==========================================="' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
