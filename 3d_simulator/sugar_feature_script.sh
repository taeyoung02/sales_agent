#!/bin/bash
set -e

# ============================================
# SuGaR Feature Field Training Pipeline
# LangSplat + CF3
# ============================================

# Helper functions
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_config() {
    echo "$1:"
    shift
    for item in "$@"; do
        echo "  - $item"
    done
}

check_yes() {
    [[ "$1" =~ ^[Yy]$ ]]
}

# ============================================
# User Input
# ============================================
print_header "SuGaR Feature Field Training Pipeline"

read -p "PROJECT_NAME (예: tesla_sugar): " PROJECT_NAME
while [[ -z "$PROJECT_NAME" ]]; do
    echo "❌ PROJECT_NAME은 필수입니다!"
    read -p "PROJECT_NAME: " PROJECT_NAME
done

# Verify that SuGaR 3DGS output exists
if [ ! -d "data/${PROJECT_NAME}/output" ]; then
    echo "❌ 오류: data/${PROJECT_NAME}/output 디렉토리를 찾을 수 없습니다!"
    echo "먼저 ./sugar_3dgs_script.sh를 실행하여 3D 재구성을 완료하세요."
    exit 1
fi

if [ ! -d "data/${PROJECT_NAME}/images" ]; then
    echo "❌ 오류: data/${PROJECT_NAME}/images 디렉토리를 찾을 수 없습니다!"
    echo "COLMAP 출력이 존재하지 않습니다. sugar_3dgs_script.sh를 먼저 실행하세요."
    exit 1
fi

# Fixed settings
VANILLA_GS_ITERATIONS=7000

echo ""
print_config "설정" \
    "PROJECT_NAME: $PROJECT_NAME" \
    "Vanilla 3DGS Iterations: $VANILLA_GS_ITERATIONS (고정)"
echo ""

read -p "진행하시겠습니까? (y/N): " CONFIRM
if ! check_yes "$CONFIRM"; then
    echo "취소되었습니다."
    exit 0
fi

export PROJECT_NAME
export VANILLA_GS_ITERATIONS

# ============================================
# Phase 4: LangSplat Feature Extraction
# ============================================
print_header "Phase 4: LangSplat Feature Extraction"

docker compose run --rm \
    -e PROJECT_NAME=${PROJECT_NAME} \
    pipeline \
    bash -c '
LANGSPLAT_DIR="/workspace/data/${PROJECT_NAME}_langsplat"

rm -rf ${LANGSPLAT_DIR}
mkdir -p ${LANGSPLAT_DIR}/images

find /workspace/data/${PROJECT_NAME}/images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) \
    -exec cp {} ${LANGSPLAT_DIR}/images/ \;

cp -r /workspace/data/${PROJECT_NAME}/sparse ${LANGSPLAT_DIR}/

cd /workspace/LangSplat

python preprocess.py --dataset_path ${LANGSPLAT_DIR} --resolution 1024 --use_dino

echo "✓ LangSplat 특징 추출 완료"
'

# ============================================
# Phase 5: CF3 Training
# ============================================
print_header "Phase 5: CF3 Feature Field Training"

docker compose run --rm \
    -e PROJECT_NAME=${PROJECT_NAME} \
    pipeline \
    bash -c '
cd /workspace/CF3
mkdir -p /workspace/data/${PROJECT_NAME}_langsplat/langsplat_features

python langsplat_feature_convert.py \
    /workspace/data/${PROJECT_NAME}_langsplat/language_features \
    /workspace/data/${PROJECT_NAME}_langsplat/langsplat_features \
    0

mkdir -p /workspace/data/${PROJECT_NAME}_cf3

python compact_feature_field.py \
    -s /workspace/data/${PROJECT_NAME}_langsplat \
    -m /workspace/data/${PROJECT_NAME}/output \
    -f langsplat \
    -o /workspace/data/${PROJECT_NAME}_cf3 \
    --resolution 0 \
    --antialiasing \
    --finetune_decoder \
    --normalize_feature \
    --iterations 1000 \
    --contrib_threshold 0.00001 \
    --alpha_threshold 0.0001 \
    --similarity_threshold 0.999 \
    --merge_grad_threshold 0.00091 \
    --no_batching

echo "✓ CF3 학습 완료"
'

# ============================================
# Pipeline Complete
# ============================================
print_header "Feature Field Training 완료!"

echo ""
echo "📁 출력 위치:"
echo "  - LangSplat: data/${PROJECT_NAME}_langsplat/"
echo "  - CF3: data/${PROJECT_NAME}_cf3/"
echo ""
echo "✓ 모든 단계 완료!"
echo ""
echo "📊 전체 출력 요약:"
echo "  - 3D 메쉬: data/${PROJECT_NAME}/mesh_textured/"
echo "  - Point Cloud: data/${PROJECT_NAME}/output/"
echo "  - Language Features: data/${PROJECT_NAME}_langsplat/"
echo "  - Compact Feature Field: data/${PROJECT_NAME}_cf3/"
echo ""
