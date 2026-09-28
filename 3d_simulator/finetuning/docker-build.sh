#!/bin/bash

# Docker 이미지 빌드 스크립트

echo "=========================================="
echo "Building SAM + CLIP Docker Image"
echo "=========================================="

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# NVIDIA Docker 런타임 확인
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime may not be configured."
    echo "Please install nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# 빌드 시작
echo ""
echo "Building Docker image..."
docker build -t sam-clip-finetuning:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "=========================================="
    echo ""
    echo "Image: sam-clip-finetuning:latest"
    echo ""
    echo "To start the container, run:"
    echo "  ./docker-run.sh"
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up -d"
    echo "  docker-compose exec sam-clip-training bash"
else
    echo ""
    echo "Build failed!"
    exit 1
fi
