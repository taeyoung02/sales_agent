#!/bin/bash

# Docker 컨테이너 실행 스크립트

echo "=========================================="
echo "Starting SAM + CLIP Training Container"
echo "=========================================="

# X11 forwarding 설정 (GUI 지원)
xhost +local:docker 2>/dev/null

# 컨테이너 실행
docker run -it --rm \
    --name sam-clip-train \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --shm-size=64gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/workspace/finetuning \
    -v $(pwd)/outputs:/workspace/finetuning/outputs \
    --network=host \
    --security-opt seccomp=unconfined \
    -w /workspace/finetuning \
    sam-clip-finetuning:latest \
    /bin/bash

# 컨테이너 종료 후 X11 권한 제거
xhost -local:docker 2>/dev/null
