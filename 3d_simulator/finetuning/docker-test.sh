#!/bin/bash

# Docker 환경 테스트 스크립트

echo "=========================================="
echo "Testing Docker Environment"
echo "=========================================="

# 이미지 존재 확인
if ! docker images | grep -q "sam-clip-finetuning"; then
    echo "Error: Docker image not found."
    echo "Please build the image first: ./docker-build.sh"
    exit 1
fi

echo "Running environment tests..."

docker run --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --shm-size=8gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/finetuning \
    -w /workspace/finetuning \
    sam-clip-finetuning:latest \
    python3 test_setup.py
