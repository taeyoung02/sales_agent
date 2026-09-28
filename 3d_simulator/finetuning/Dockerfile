# SAM + CLIP Fine-tuning Docker Image
# Base: NVIDIA PyTorch Container (CUDA 13.0, PyTorch 2.9, Python 3.12)
# Compatible with NVIDIA GB10 (sm_121)
FROM nvcr.io/nvidia/pytorch:25.09-py3 AS base

# 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul

# CUDA Environment (CUDA 13.0 supports sm_89, sm_90, sm_121)
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.1"
ENV FORCE_CUDA=1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Fix HPC-X/UCX library conflict on aarch64
ENV LD_PRELOAD=/opt/hpcx/ucx/lib/libucm.so.0:/opt/hpcx/ucx/lib/libuct.so.0:/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0
ENV OMPI_MCA_coll_hcoll_enable=0
ENV HCOLL_ENABLE_MCAST_ALL=0

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# NGC container already has PyTorch 2.9 with CUDA support
# Just install additional required packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Segment Anything 설치
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# 추가 유틸리티
RUN pip install --no-cache-dir jupyter ipython

# 포트 노출 (TensorBoard용)
EXPOSE 6006

# 작업 디렉토리 설정
WORKDIR /workspace/finetuning

# 기본 명령어
CMD ["/bin/bash"]
