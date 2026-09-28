#!/bin/bash

# TensorBoard 실행 스크립트 (Docker 컨테이너 내부)

echo "=========================================="
echo "Starting TensorBoard in Docker"
echo "=========================================="

# 로그 디렉토리 확인
LOG_DIR=${1:-"./outputs"}

echo "Monitoring logs in: $LOG_DIR"
echo "TensorBoard will be available at: http://localhost:6006"
echo ""

docker run --rm -d \
    --name sam-clip-tensorboard \
    -p 6006:6006 \
    -v $(pwd):/workspace/finetuning \
    -w /workspace/finetuning \
    sam-clip-finetuning:latest \
    tensorboard --logdir=$LOG_DIR --host=0.0.0.0 --port=6006

echo "TensorBoard started!"
echo ""
echo "To view, open: http://localhost:6006"
echo "To stop: docker stop sam-clip-tensorboard"
