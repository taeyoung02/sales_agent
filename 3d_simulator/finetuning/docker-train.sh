#!/bin/bash

# Docker 컨테이너에서 CLIP 학습 시작 스크립트
# 기본적으로 파인튜닝된 SAM 모델 사용

echo "=========================================="
echo "Starting CLIP Training in Docker Container"
echo "=========================================="

# Configuration - Default to using fine-tuned SAM
USE_FINETUNED_SAM=${USE_FINETUNED_SAM:-true}
FINETUNED_SAM_DIR=${FINETUNED_SAM_DIR:-"outputs/sam_20260101_003941"}

# Build command with optional fine-tuned SAM
TRAIN_ARGS="--data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --sam-model-type vit_b \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --batch-size 4 \
    --num-workers 8 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --save-dir ./outputs \
    --save-every 5 \
    --device cuda \
    --gpu-id 0 \
    --use-amp \
    --accumulation-steps 2"

# Add fine-tuned SAM arguments if enabled
if [ "$USE_FINETUNED_SAM" = "true" ]; then
    echo "✓ Using Fine-tuned SAM Model"
    echo "  Directory: $FINETUNED_SAM_DIR"
    TRAIN_ARGS="$TRAIN_ARGS --use-finetuned-sam --finetuned-sam-dir $FINETUNED_SAM_DIR"
else
    echo "✗ Using Pretrained SAM (not recommended)"
fi

# docker-compose 사용
if command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    
    # 컨테이너 시작
    docker-compose up -d
    
    # 학습 실행
    docker-compose exec sam-clip-training python3 train.py $TRAIN_ARGS
else
    echo "docker-compose not found. Using docker run..."
    
    docker run --rm \
        --device nvidia.com/gpu=all \
        --ipc=host \
        --shm-size=64gb \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd):/workspace/finetuning \
        -v $(pwd)/outputs:/workspace/finetuning/outputs \
        --network=host \
        -w /workspace/finetuning \
        sam-clip-finetuning:latest \
        python3 train.py $TRAIN_ARGS
fi
