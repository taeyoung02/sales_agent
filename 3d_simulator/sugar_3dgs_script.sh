#!/bin/bash
set -e

# ============================================
# SuGaR 3D Reconstruction Pipeline
# COLMAP + Vanilla 3DGS + SuGaR + Mesh Extraction
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
print_header "SuGaR 3D Gaussian Splatting Pipeline"

read -p "PROJECT_NAME (예: tesla_sugar): " PROJECT_NAME
while [[ -z "$PROJECT_NAME" ]]; do
    echo "❌ PROJECT_NAME은 필수입니다!"
    read -p "PROJECT_NAME: " PROJECT_NAME
done

read -p "이미지 경로 (예: ./my_images): " IMAGE_PATH
while [[ ! -d "$IMAGE_PATH" ]]; do
    echo "❌ 경로를 찾을 수 없습니다: $IMAGE_PATH"
    read -p "이미지 경로: " IMAGE_PATH
done

read -p "빛번짐/반사 제거 적용? (Y/n): " USE_ESRGAN_PREPROCESS
USE_ESRGAN_PREPROCESS=${USE_ESRGAN_PREPROCESS:-Y}

read -p "자동 배경 제거? (Y/n): " AUTO_REMOVE_BACKGROUND
AUTO_REMOVE_BACKGROUND=${AUTO_REMOVE_BACKGROUND:-Y}

# SuGaR specific settings
read -p "메쉬 정점 수 (기본: 1000000, 저사양: 200000): " N_VERTICES
N_VERTICES=${N_VERTICES:-1000000}

read -p "Triangle당 Gaussian 수 (기본: 1, 고품질: 6): " GAUSSIANS_PER_TRIANGLE
GAUSSIANS_PER_TRIANGLE=${GAUSSIANS_PER_TRIANGLE:-1}

read -p "Refinement 시간 (short/medium/long, 기본: medium): " REFINEMENT_TIME
REFINEMENT_TIME=${REFINEMENT_TIME:-medium}

read -p "UV 텍스처 메쉬 내보내기? (Y/n): " EXPORT_TEXTURED_MESH
EXPORT_TEXTURED_MESH=${EXPORT_TEXTURED_MESH:-Y}

read -p "메쉬 재구성 알고리즘 (poisson/alpha, 기본: poisson): " MESH_ALGORITHM
MESH_ALGORITHM=${MESH_ALGORITHM:-poisson}

# Fixed settings
VANILLA_GS_ITERATIONS=7000
MAX_WIDTH=3240

echo ""
print_config "설정" \
    "PROJECT_NAME: $PROJECT_NAME" \
    "IMAGE_PATH: $IMAGE_PATH" \
    "Vanilla 3DGS Iterations: $VANILLA_GS_ITERATIONS (고정)" \
    "빛번짐/반사 제거: $USE_ESRGAN_PREPROCESS" \
    "자동 배경 제거: $AUTO_REMOVE_BACKGROUND" \
    "메쉬 정점 수: $N_VERTICES" \
    "Triangle당 Gaussian: $GAUSSIANS_PER_TRIANGLE" \
    "Refinement 시간: $REFINEMENT_TIME" \
    "메쉬 재구성 알고리즘: $MESH_ALGORITHM" \
    "UV 텍스처 메쉬: $EXPORT_TEXTURED_MESH"
echo ""

read -p "진행하시겠습니까? (y/N): " CONFIRM
if ! check_yes "$CONFIRM"; then
    echo "취소되었습니다."
    exit 0
fi



export PROJECT_NAME
export N_VERTICES
export GAUSSIANS_PER_TRIANGLE
export REFINEMENT_TIME
export VANILLA_GS_ITERATIONS
export EXPORT_TEXTURED_MESH
export MESH_ALGORITHM

# ============================================
# Phase 0: Auto Image Enhancement
# ============================================
print_header "Phase 0: Image Preprocessing"

mkdir -p data/${PROJECT_NAME}/input
cp ${IMAGE_PATH}/* data/${PROJECT_NAME}/input/

# Detect image resolution and apply RealESRGAN if needed
SAMPLE_IMAGE=$(find data/${PROJECT_NAME}/input -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) | head -1)
if [[ -n "$SAMPLE_IMAGE" ]]; then
    IMAGE_WIDTH=$(identify -format "%w" "$SAMPLE_IMAGE" 2>/dev/null || echo "0")
    
    if [[ "$IMAGE_WIDTH" -lt 1080 && "$IMAGE_WIDTH" -gt 0 ]]; then
        print_header "Phase 0.1: RealESRGAN Image Enhancement (Auto)"
        echo "✓ 이미지 해상도가 ${IMAGE_WIDTH}px로 1080px 미만입니다. RealESRGAN 적용..."

        docker run --rm \
            --device nvidia.com/gpu=all \
            -v $(pwd)/data/${PROJECT_NAME}/input:/app/inputs \
            -v $(pwd)/data/${PROJECT_NAME}/enhanced:/app/outputs \
            -v $(pwd)/Real-ESRGAN-0.3.0/weights:/app/weights \
            -e MODEL_NAME=RealESRNet_x4plus \
            -e OUTSCALE=4 \
            -e TILE=800 \
            realesrgan-preprocessing:latest  # Backup and replace
        mkdir -p data/${PROJECT_NAME}/input_original
        mv data/${PROJECT_NAME}/input/* data/${PROJECT_NAME}/input_original/
        cp -r data/${PROJECT_NAME}/enhanced/* data/${PROJECT_NAME}/input/
        
        INPUT_COUNT=$(find data/${PROJECT_NAME}/input -type f | wc -l)
        echo "✓ RealESRGAN 완료: ${INPUT_COUNT}개 파일"
    else
        echo "✓ 이미지 해상도 충분 (${IMAGE_WIDTH}px). RealESRGAN 건너뛰기"
    fi
fi

# ============================================
# Phase 0.2: ESRGAN Light Reflection & Bloom Removal
# ============================================
if check_yes "$USE_ESRGAN_PREPROCESS"; then
    print_header "Phase 0.2: ESRGAN 빛번짐/반사 제거"
    
    mkdir -p data/${PROJECT_NAME}/esrgan_temp
    cp data/${PROJECT_NAME}/input/* data/${PROJECT_NAME}/esrgan_temp/
    
    docker run --rm \
        --device nvidia.com/gpu=all \
        -v $(pwd)/esrgan_preprocessing.py:/workspace/esrgan_preprocessing.py \
        -v $(pwd)/data:/workspace/data \
        --ipc=host \
        --shm-size=64gb \
        3d-reconstruction-pipeline:latest \
        bash -c "
python /workspace/esrgan_preprocessing.py \
    --images_dir /workspace/data/${PROJECT_NAME}/esrgan_temp \
    --output_dir /workspace/data/${PROJECT_NAME}/input \
    --remove_reflection \
    --max_width ${MAX_WIDTH} \
    --device cuda
"
    
    rm -rf data/${PROJECT_NAME}/esrgan_temp
    
    INPUT_COUNT=$(find data/${PROJECT_NAME}/input -type f | wc -l)
    echo "✓ ESRGAN 전처리 완료: ${INPUT_COUNT}개 파일"
fi

# ============================================
# Phase 1: COLMAP
# ============================================
print_header "Phase 1: COLMAP Camera Pose Estimation"

docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    3d-reconstruction-pipeline:latest \
    bash -c '
cd /workspace/data/${PROJECT_NAME}

rm -f database.db
rm -rf sparse distorted dense

colmap feature_extractor \
    --database_path database.db \
    --image_path input \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.num_threads 8 \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192 \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 0 \
    --SiftMatching.num_threads 8

mkdir -p sparse
colmap mapper \
    --database_path database.db \
    --image_path input \
    --output_path sparse 

if [ ! -d sparse/0 ]; then
    mkdir -p sparse/0
    mv sparse/*.bin sparse/0/ 2>/dev/null || true
    mv sparse/*.txt sparse/0/ 2>/dev/null || true
fi

if [ ! -f sparse/0/cameras.bin ]; then
    echo "ERROR: Reconstruction failed"
    exit 1
fi

colmap image_undistorter \
    --image_path input \
    --input_path sparse/0 \
    --output_path dense \
    --output_type COLMAP

if [ ! -d dense/sparse ]; then
    echo "ERROR: Undistortion failed"
    exit 1
fi

mkdir -p images
cp -r dense/images/* images/ 2>/dev/null || cp input/* images/
cp dense/sparse/*.bin sparse/0/
cp dense/sparse/*.txt sparse/0/ 2>/dev/null || true

echo "✓ COLMAP 완료: $(ls images/ | wc -l)개 이미지"
'

# ============================================
# Phase 2: Rembg Masking (for SuGaR)
# ============================================

print_header "Phase 2: Rembg 배경 제거 (SuGaR용)"
    
docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/generate_rembg_masks.py:/app/generate_rembg_masks.py \
    --ipc=host \
    --shm-size=64gb \
    --entrypoint python \
    rembg-cuda:latest \
    /app/generate_rembg_masks.py \
    --images_dir /app/data/${PROJECT_NAME}/images \
    --output_dir /app/data/${PROJECT_NAME}/masks \
    --model isnet-general-use \
    --device cuda \
    --output_type mask \
    --alpha_matting \
    --alpha_matting_foreground_threshold 240 \
    --alpha_matting_background_threshold 10 \
    --alpha_matting_erode_size 10

MASK_OPTION="--masks masks"
echo "✓ 배경 제거 마스크 생성 완료"

# ============================================
# Phase 3: SuGaR Training
# ============================================
print_header "Phase 3: SuGaR Training (Surface-Aligned Gaussian Splatting)"

# Convert REFINEMENT_TIME to iterations
case "$REFINEMENT_TIME" in
    "short")
        REFINEMENT_ITERATIONS=2000
        ;;
    "medium")
        REFINEMENT_ITERATIONS=7000
        ;;
    "long")
        REFINEMENT_ITERATIONS=15000
        ;;
    *)
        REFINEMENT_ITERATIONS=2000
        ;;
esac

docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/SuGaR-main/sugar_extractors/texture.py:/workspace/SuGaR/sugar_extractors/texture.py \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    -e VANILLA_GS_ITERATIONS=${VANILLA_GS_ITERATIONS} \
    -e N_VERTICES=${N_VERTICES} \
    -e GAUSSIANS_PER_TRIANGLE=${GAUSSIANS_PER_TRIANGLE} \
    -e REFINEMENT_ITERATIONS=${REFINEMENT_ITERATIONS} \
    -e EXPORT_TEXTURED_MESH=${EXPORT_TEXTURED_MESH} \
    -e MESH_ALGORITHM=${MESH_ALGORITHM} \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    sugar-meshing:latest \
    bash -c '
cd /workspace/SuGaR

# # Step 1: Train Vanilla 3DGS
# echo "===== Step 1/5: Training Vanilla 3DGS ====="
# /opt/conda/envs/sugar/bin/python gaussian_splatting/train.py \
#     -s /workspace/data/${PROJECT_NAME}/ \
#     -m /workspace/data/${PROJECT_NAME}/output_vanilla/ \
#     ${MASK_OPTION} \
#     --iterations ${VANILLA_GS_ITERATIONS} \
#     --save_iterations ${VANILLA_GS_ITERATIONS} \
#     --checkpoint_iterations ${VANILLA_GS_ITERATIONS}

# Step 2: Train Coarse SuGaR
echo ""
echo "===== Step 2/5: Training Coarse SuGaR ====="
/opt/conda/envs/sugar/bin/python train_coarse_density.py \
    -s /workspace/data/${PROJECT_NAME}/ \
    -c /workspace/data/${PROJECT_NAME}/output_vanilla/ \
    -i ${VANILLA_GS_ITERATIONS} \
    -o /workspace/data/${PROJECT_NAME}/output_coarse/

# Step 3: Extract Coarse Mesh
echo ""
echo "===== Step 3/5: Extracting Coarse Mesh ====="
COARSE_MODEL=$(find /workspace/data/${PROJECT_NAME}/output_coarse -type f -name "*.pt" | head -1)

/opt/conda/envs/sugar/bin/python extract_mesh.py \
    -s /workspace/data/${PROJECT_NAME}/ \
    -c /workspace/data/${PROJECT_NAME}/output_vanilla/ \
    -m ${COARSE_MODEL} \
    -i ${VANILLA_GS_ITERATIONS} \
    -o /workspace/data/${PROJECT_NAME}/mesh_coarse/ \
    -d ${N_VERTICES} \
    --enable_mask_culling True \
    --mask_voting_threshold 0.75 \
    --mask_dilation_kernel 7 \
    --debug_culling True

# Step 4: Refine SuGaR
echo ""
echo "===== Step 4/5: Refining SuGaR ====="
MESH_PATH=$(find /workspace/data/${PROJECT_NAME}/mesh_coarse -name "*.ply" | head -1)
if [ -z "${MESH_PATH}" ]; then
    echo "❌ Error: No mesh file found in mesh_coarse directory"
    exit 1
fi
echo "Using mesh: ${MESH_PATH}"

/opt/conda/envs/sugar/bin/python train_refined.py \
    -s /workspace/data/${PROJECT_NAME}/ \
    -c /workspace/data/${PROJECT_NAME}/output_vanilla/ \
    -m ${MESH_PATH} \
    -i ${VANILLA_GS_ITERATIONS} \
    -o /workspace/data/${PROJECT_NAME}/output_refined/ \
    -g ${GAUSSIANS_PER_TRIANGLE} \
    -f ${REFINEMENT_ITERATIONS}

# Step 5: Export Textured Mesh
if [[ "${EXPORT_TEXTURED_MESH}" =~ ^[Yy]$ ]]; then
    echo ""
    echo "===== Step 5/5: Exporting UV Textured Mesh ====="
    REFINED_MODEL=$(find /workspace/data/${PROJECT_NAME}/output_refined -type f -name "*.pt" | head -1)
    if [ -z "${REFINED_MODEL}" ]; then
        echo "❌ Error: No refined model file found"
        exit 1
    fi
    echo "Using refined model: ${REFINED_MODEL}"
    
    # Create expected coarse mesh directory structure
    SCENE_NAME=$(basename /workspace/data/${PROJECT_NAME})
    mkdir -p /workspace/SuGaR/output/coarse_mesh/${SCENE_NAME}
    
    # Copy coarse mesh to expected location with correct filename
    COARSE_MESH=$(find /workspace/data/${PROJECT_NAME}/mesh_coarse -name "*.ply" | head -1)
    if [ -z "${COARSE_MESH}" ]; then
        echo "❌ Error: No coarse mesh file found"
        exit 1
    fi
    
    # Extract expected filename from refined model path
    REFINED_DIR=$(basename $(dirname ${REFINED_MODEL}))
    EXPECTED_MESH_NAME=$(echo ${REFINED_DIR} | sed "s/_normalconsistency.*//g" | sed "s/sugarfine/sugarmesh/g").ply
    TARGET_MESH_PATH=/workspace/SuGaR/output/coarse_mesh/${SCENE_NAME}/${EXPECTED_MESH_NAME}
    
    cp ${COARSE_MESH} ${TARGET_MESH_PATH}
    echo "✓ Coarse mesh copied to: ${TARGET_MESH_PATH}"
    
    # Use smaller square_size to reduce memory usage and prevent bin overflow
    /opt/conda/envs/sugar/bin/python extract_refined_mesh_with_texture.py \
        -s /workspace/data/${PROJECT_NAME}/ \
        -c /workspace/data/${PROJECT_NAME}/output_vanilla/ \
        -m ${REFINED_MODEL} \
        -i ${VANILLA_GS_ITERATIONS} \
        -o /workspace/data/${PROJECT_NAME}/mesh_textured/ \
        -n ${GAUSSIANS_PER_TRIANGLE}
fi
'

# ============================================
# Pipeline Complete
# ============================================
print_header "SuGaR 3D Reconstruction 완료!"

echo ""
echo "📁 출력 위치:"
echo "  - Vanilla 3DGS: data/${PROJECT_NAME}/output_vanilla/"
echo "  - Coarse SuGaR: data/${PROJECT_NAME}/output_coarse/"
echo "  - Refined SuGaR: data/${PROJECT_NAME}/output_refined/"
echo "  - Coarse Mesh: data/${PROJECT_NAME}/mesh_coarse/"
if check_yes "$EXPORT_TEXTURED_MESH"; then
    echo "  - Textured Mesh: data/${PROJECT_NAME}/mesh_textured/"
fi
echo "  - CF3 호환 출력: data/${PROJECT_NAME}/output/"
echo ""
echo "✓ 3D 메쉬 생성 완료!"
echo ""
echo "📌 다음 단계:"
echo "  Feature Field 학습을 진행하려면:"
echo "  ./sugar_feature_script.sh 실행"
echo ""
