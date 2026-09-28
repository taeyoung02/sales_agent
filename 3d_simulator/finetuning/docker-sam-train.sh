#!/bin/bash
#
# SAM Fine-tuning Training Script (Docker)
# Fine-tune SAM mask decoder and prompt encoder on VehicleSeg10K
#

set -e

# Configuration
IMAGE_NAME="3d-reconstruction-pipeline:latest"
CONTAINER_NAME="sam-finetuning"

# Default arguments
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_MASKS_PER_IMAGE=${MAX_MASKS_PER_IMAGE:-10}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VALID_SAMPLES=${MAX_VALID_SAMPLES:-}

# Get the absolute path to the finetuning directory
FINETUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "SAM Fine-tuning Training"
echo "================================================"
echo "Docker Image: $IMAGE_NAME"
echo "Container: $CONTAINER_NAME"
echo "Finetuning Dir: $FINETUNING_DIR"
echo ""
echo "Training Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Learning Rate: $LR"
echo "  - Workers: $NUM_WORKERS"
echo "  - Max Masks/Image: $MAX_MASKS_PER_IMAGE"
[ -n "$MAX_TRAIN_SAMPLES" ] && echo "  - Max Train Samples: $MAX_TRAIN_SAMPLES"
[ -n "$MAX_VALID_SAMPLES" ] && echo "  - Max Valid Samples: $MAX_VALID_SAMPLES"
echo "================================================"
echo ""

# Check if Docker image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Error: Docker image '$IMAGE_NAME' not found."
    echo "Please build the image first using docker-build.sh"
    exit 1
fi

# Remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
fi

# Prepare training command
TRAIN_CMD="python sam_train.py \
    --dataset_root /workspace/finetuning/VehicleSeg10K \
    --sam_checkpoint /workspace/finetuning/checkpoints/sam_vit_b_01ec64.pth \
    --sam_model_type vit_b \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --focal_weight 1.0 \
    --dice_weight 1.0 \
    --iou_weight 0.5 \
    --num_prompts 5 \
    --max_masks_per_image $MAX_MASKS_PER_IMAGE \
    --use_bbox \
    --use_point \
    --use_amp \
    --output_dir /workspace/finetuning/outputs"

# Add optional arguments
[ -n "$MAX_TRAIN_SAMPLES" ] && TRAIN_CMD="$TRAIN_CMD --max_train_samples $MAX_TRAIN_SAMPLES"
[ -n "$MAX_VALID_SAMPLES" ] && TRAIN_CMD="$TRAIN_CMD --max_valid_samples $MAX_VALID_SAMPLES"

echo "Running SAM fine-tuning training..."
echo ""

# Run Docker container with GPU support
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --shm-size=64gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$FINETUNING_DIR:/workspace/finetuning" \
    --network=host \
    -w /workspace/finetuning \
    "$IMAGE_NAME" \
    bash -c "$TRAIN_CMD"

echo ""
echo "================================================"
echo "SAM fine-tuning completed!"
echo "Output saved to: $FINETUNING_DIR/outputs"
echo "================================================"
