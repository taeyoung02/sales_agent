#!/bin/bash

# SAM + CLIP 디버깅 스크립트
# 학습된 모델로 샘플 이미지들을 분석하고 시각화합니다.

echo "=========================================="
echo "SAM + CLIP Debugging & Visualization"
echo "=========================================="

# 기본 설정
CHECKPOINT=${1:-"outputs/sam_clip_20260101_220944/best_model.pth"}
SAM_CHECKPOINT=${2:-"outputs/sam_20260101_003941/best_model.pth"}
DATA_ROOT=${3:-"./VehicleSeg10K"}
OUTPUT_DIR=${4:-"debug_output"}

echo ""
echo "Configuration:"
echo "  - Model checkpoint: $CHECKPOINT"
echo "  - SAM checkpoint: $SAM_CHECKPOINT"
echo "  - Dataset root: $DATA_ROOT"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Docker에서 실행
docker run --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --shm-size=8gb \
    -v $(pwd):/workspace/finetuning \
    -w /workspace/finetuning \
    sam-clip-finetuning:latest \
    bash -c "
# 검증 데이터셋에서 랜덤하게 5개 샘플 선택
VALID_IMAGES=\$(find ${DATA_ROOT}/images/valid -type f \( -name '*.jpg' -o -name '*.png' \) | shuf -n 5)

echo 'Selected sample images:'
echo \"\$VALID_IMAGES\" | nl

echo ''
echo 'Running debugging pipeline...'
echo ''

# 각 이미지에 대해 디버깅 실행
for img in \$VALID_IMAGES; do
    python3 debugging.py \\
        --image \"\$img\" \\
        --checkpoint ${CHECKPOINT} \\
        --sam-checkpoint ${SAM_CHECKPOINT} \\
        --data-root ${DATA_ROOT} \\
        --output-dir ${OUTPUT_DIR}
    
    echo ''
done

echo '=========================================='
echo 'Debugging Complete!'
echo '=========================================='
echo ''
echo 'Visualizations saved to: ${OUTPUT_DIR}/'
echo ''
echo 'Output files per image:'
echo '  1. *_1_sam_masks.png           - SAM mask generation results'
echo '  2. *_2_context_aware_cropping.png - Context-aware blur comparison'
echo '  3. *_3_clip_predictions.png    - CLIP classification results'
echo '  4. *_4_gt_comparison.png       - Ground truth comparison'
echo ''
"

echo ""
echo "✓ Debugging complete! Check the output directory:"
echo "  $(pwd)/${OUTPUT_DIR}/"
echo ""
