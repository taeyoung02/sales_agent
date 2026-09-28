#!/bin/bash

# Quick start training script

echo "Starting SAM + CLIP training on VehicleSeg10K..."

python train.py \
    --data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --sam-model-type vit_b \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --batch-size 2 \
    --num-workers 4 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --save-dir ./outputs \
    --save-every 5 \
    --device cuda \
    --gpu-id 0
