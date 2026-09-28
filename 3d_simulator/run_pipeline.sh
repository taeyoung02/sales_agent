#!/bin/bash
set -e

# ============================================
# 3D Reconstruction Full Pipeline Runner
# (SuGaR Vanilla 3DGS + LangSplat + CF3)
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
print_header "3D Reconstruction Pipeline"

read -p "PROJECT_NAME (예: tesla_new): " PROJECT_NAME
while [[ -z "$PROJECT_NAME" ]]; do
    echo "❌ PROJECT_NAME은 필수입니다!"
    read -p "PROJECT_NAME: " PROJECT_NAME
done

read -p "비디오에서 이미지 추출? (y/N): " EXTRACT_FROM_VIDEO
EXTRACT_FROM_VIDEO=${EXTRACT_FROM_VIDEO:-N}

if check_yes "$EXTRACT_FROM_VIDEO"; then
    # 이미지 경로 검증 건너뛰기 (비디오에서 추출할 것임)
    IMAGE_PATH="dummy"
    read -p "비디오 파일 경로 (예: ./videos/game.mp4): " VIDEO_PATH
    while [[ ! -f "$VIDEO_PATH" ]]; do
        echo "❌ 비디오 파일을 찾을 수 없습니다: $VIDEO_PATH"
        read -p "비디오 파일 경로: " VIDEO_PATH
    done
fi


if ! check_yes "$EXTRACT_FROM_VIDEO"; then
    read -p "이미지 경로 (예: ./my_images): " IMAGE_PATH
    while [[ ! -d "$IMAGE_PATH" ]]; do
        echo "❌ 경로를 찾을 수 없습니다: $IMAGE_PATH"
        read -p "이미지 경로: " IMAGE_PATH
    done
fi

read -p "빛번짐/반사 제거 적용? (Y/n): " USE_ESRGAN_PREPROCESS
USE_ESRGAN_PREPROCESS=${USE_ESRGAN_PREPROCESS:-Y}

read -p "자동 배경 제거? (Y/n): " AUTO_REMOVE_BACKGROUND
AUTO_REMOVE_BACKGROUND=${AUTO_REMOVE_BACKGROUND:-Y}

read -p "디버그 모드 (중간 파일 유지)? (y/N): " DEBUG_MODE
DEBUG_MODE=${DEBUG_MODE:-N}

# Fixed settings
ITERATIONS=15000
MAX_WIDTH=3240
FRAME_INTERVAL=30

echo ""
print_config "설정" \
    "PROJECT_NAME: $PROJECT_NAME" \
    "IMAGE_PATH: $IMAGE_PATH" \
    "ITERATIONS: $ITERATIONS (고정)" \
    "빛번짐/반사 제거: $USE_ESRGAN_PREPROCESS" \
    "자동 배경 제거: $AUTO_REMOVE_BACKGROUND" \
    "비디오 추출: $EXTRACT_FROM_VIDEO" \
    $(check_yes "$EXTRACT_FROM_VIDEO" && echo "비디오 경로: $VIDEO_PATH" || echo "") \
    $(check_yes "$EXTRACT_FROM_VIDEO" && echo "프레임 간격: $FRAME_INTERVAL" || echo "") \
    "디버그 모드: $DEBUG_MODE"
echo ""

read -p "진행하시겠습니까? (y/N): " CONFIRM
if ! check_yes "$CONFIRM"; then
    echo "취소되었습니다."
    exit 0
fi

export PROJECT_NAME
export ITERATIONS

# ============================================
# Phase -1: Video to Images (Optional)
# ============================================
if check_yes "$EXTRACT_FROM_VIDEO"; then
    print_header "Phase -1: 비디오에서 이미지 추출 (katna_mov_to_img.py)"
    
    # 절대 경로로 변환
    VIDEO_PATH_ABS=$(realpath "$VIDEO_PATH")
    OUTPUT_DIR_ABS=$(pwd)/data/${PROJECT_NAME}/video_extracted
    
    echo "📹 비디오: $VIDEO_PATH_ABS"
    echo "📁 출력 폴더: $OUTPUT_DIR_ABS"
    echo "⏱️  간격: ${FRAME_INTERVAL} 프레임"
    
    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_DIR_ABS"
    
    # Docker 컨테이너에서 실행
    docker run --rm \
        --device nvidia.com/gpu=all \
        -v "$VIDEO_PATH_ABS:/workspace/input_video.mp4:ro" \
        -v "$OUTPUT_DIR_ABS:/workspace/output" \
        -v $(pwd)/utils/katna_mov_to_img.py:/workspace/katna_mov_to_img.py \
        --ipc=host \
        --shm-size=64gb \
        3d-reconstruction-pipeline:latest \
        bash -c "
python /workspace/katna_mov_to_img.py /workspace/input_video.mp4 /workspace/output --interval ${FRAME_INTERVAL}
RESULT=\$?
"
    
    DOCKER_RESULT=$?
    
    if [ $DOCKER_RESULT -eq 0 ]; then
        TOTAL_EXTRACTED=$(find "$OUTPUT_DIR_ABS" -type f -name "*.jpg" | wc -l)
        echo ""
        echo "✅ 비디오 추출 완료! ${TOTAL_EXTRACTED}개 이미지"
        echo ""
        
        # 추출된 이미지를 IMAGE_PATH로 자동 사용
        IMAGE_PATH="$OUTPUT_DIR_ABS"
        echo "✓ IMAGE_PATH 자동 업데이트: $IMAGE_PATH"
    else
        echo ""
        echo "❌ 비디오 추출 실패"
        exit 1
    fi
fi

# ============================================
# Phase 0: Auto Image Enhancement
# ============================================
print_header "Phase 0: Image Preprocessing"

mkdir -p data/${PROJECT_NAME}/input

# 이미지 복사 (비디오 추출 or 기존 이미지)
if check_yes "$EXTRACT_FROM_VIDEO"; then
    # 비디오에서 추출한 경우: video_extracted → input
    echo "✓ 비디오 추출 이미지를 input으로 복사 중..."
    COPIED_COUNT=$(find data/${PROJECT_NAME}/video_extracted -type f -name "*.jpg" -exec cp {} data/${PROJECT_NAME}/input/ \; -print | wc -l)
    echo "✓ ${COPIED_COUNT}개 파일 복사 완료"
else
    # 기존 이미지 폴더에서 복사
    echo "✓ 이미지 폴더에서 복사 중..."
    cp ${IMAGE_PATH}/* data/${PROJECT_NAME}/input/
    COPIED_COUNT=$(find data/${PROJECT_NAME}/input -type f | wc -l)
    echo "✓ ${COPIED_COUNT}개 파일 복사 완료"
fi

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
            realesrgan-preprocessing:latest
        
        # Backup and replace
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
    -e AUTO_REMOVE_BACKGROUND=${AUTO_REMOVE_BACKGROUND} \
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
# Phase 2: Rembg Masking (for 3DGS)
# ============================================
if check_yes "$AUTO_REMOVE_BACKGROUND"; then
    print_header "Phase 2: Rembg 배경 제거 (3DGS용)"
    
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
else
    MASK_OPTION=""
    echo "✓ 배경 제거 건너뛰기"
fi

# ============================================
# Phase 3: SuGaR Vanilla 3D Gaussian Splatting
# ============================================
print_header "Phase 3: SuGaR Vanilla 3D Gaussian Splatting Training"

# Train Vanilla 3DGS with SuGaR
docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    -e ITERATIONS=${ITERATIONS} \
    -e MASK_OPTION="${MASK_OPTION}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    sugar-meshing:latest \
    bash -c "
cd /workspace/SuGaR

/opt/conda/envs/sugar/bin/python gaussian_splatting/train.py \
    -s /workspace/data/${PROJECT_NAME} \
    -m /workspace/data/${PROJECT_NAME}/output \
    ${MASK_OPTION} \
    --iterations ${ITERATIONS} \
    --checkpoint_iterations ${ITERATIONS} \
    --sh_degree 3 \
    --densify_until_iter 12000

echo '✓ SuGaR Vanilla 3DGS 학습 완료'
"

# ============================================
# Phase 3.5: Point Cloud Pruning
# ============================================
print_header "Phase 3.5: Point Cloud Pruning"

docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/mahalanobis_pruning.py:/workspace/mahalanobis_pruning.py \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    -e ITERATIONS=${ITERATIONS} \
    3d-reconstruction-pipeline:latest \
    bash -c '
cd /workspace

ITER_DIR="/workspace/data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}"
INPUT_PLY="${ITER_DIR}/point_cloud.ply"
BACKUP_PLY="${ITER_DIR}/point_cloud_original.ply"
OUTPUT_PLY="${ITER_DIR}/point_cloud_pruned.ply"
COLMAP_PATH="/workspace/data/${PROJECT_NAME}/sparse/0"
IMAGE_FOLDER="/workspace/data/${PROJECT_NAME}/images"

cp ${INPUT_PLY} ${BACKUP_PLY}

python mahalanobis_pruning.py \
    --input ${INPUT_PLY} \
    --output ${OUTPUT_PLY} \
    -n 100 \
    --colmap-path ${COLMAP_PATH} \
    --image-folder ${IMAGE_FOLDER} \
    --convergence 0.0001

mv ${OUTPUT_PLY} ${INPUT_PLY}

echo "✓ Point Cloud 정리 완료"
'

# ============================================
# Phase 4: LangSplat Feature Extraction
# ============================================
print_header "Phase 4: LangSplat Feature Extraction"

docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/LangSplat:/workspace/LangSplat \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    3d-reconstruction-pipeline:latest \
    bash -c '
LANGSPLAT_DIR="/workspace/data/${PROJECT_NAME}_langsplat"

rm -rf ${LANGSPLAT_DIR}
mkdir -p ${LANGSPLAT_DIR}/images

# Copy all common image formats including .jpeg
find /workspace/data/${PROJECT_NAME}/images -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPG" -o -name "*.JPEG" -o -name "*.PNG" \) \
    -exec cp {} ${LANGSPLAT_DIR}/images/ \;

IMAGE_COUNT=$(find ${LANGSPLAT_DIR}/images -type f | wc -l)
echo "✓ Copied ${IMAGE_COUNT} images to LangSplat directory"

if [ "${IMAGE_COUNT}" -eq 0 ]; then
    echo "❌ ERROR: No images found in /workspace/data/${PROJECT_NAME}/images"
    exit 1
fi

cp -r /workspace/data/${PROJECT_NAME}/sparse ${LANGSPLAT_DIR}/

cd /workspace/LangSplat

python preprocess.py --dataset_path ${LANGSPLAT_DIR} --resolution 1024 --use_dino

echo "✓ LangSplat 특징 추출 완료"
'

# ============================================
# Phase 5: CF3 Training
# ============================================
print_header "Phase 5: CF3 Feature Field Training"

docker run --rm \
    --device nvidia.com/gpu=all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/CF3:/workspace/CF3 \
    --ipc=host \
    --shm-size=64gb \
    -e PROJECT_NAME=${PROJECT_NAME} \
    3d-reconstruction-pipeline:latest \
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
# Pipeline Complete & Cleanup
# ============================================
print_header "파이프라인 완료!"

if ! check_yes "$DEBUG_MODE"; then
    print_header "최종 파일 정리 중..."
    
    # 필수 파일 경로 정의
    REQUIRED_FILES=(
        "data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
        "data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}/point_cloud_pruned_pose.json"
        "data/${PROJECT_NAME}_cf3/autoencoder.pth"
        "data/${PROJECT_NAME}_cf3/feature_field.ply"
    )
    
    # 파일 존재 여부 확인
    MISSING_FILES=0
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "⚠️  경고: $file 파일을 찾을 수 없습니다"
            MISSING_FILES=$((MISSING_FILES + 1))
        fi
    done
    
    if [ $MISSING_FILES -gt 0 ]; then
        echo "❌ $MISSING_FILES 개의 필수 파일이 없습니다."
        echo "⚠️  디버그를 위해 모든 파일을 유지합니다."
        echo ""
        echo "📁 출력 위치 (디버그 모드 - 파일 누락):"
        echo "  - 3DGS: data/${PROJECT_NAME}/output/"
        echo "  - LangSplat: data/${PROJECT_NAME}_langsplat/"
        echo "  - CF3: data/${PROJECT_NAME}_cf3/"
    else
        # 출력 디렉토리 생성
        mkdir -p data/${PROJECT_NAME}_output
        
        # 필수 파일 4개 복사
        echo "✓ 필수 파일 저장 중..."
        cp data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply \
           data/${PROJECT_NAME}_output/point_cloud.ply
        echo "  → point_cloud.ply"
        
        cp data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}/point_cloud_pruned_pose.json \
           data/${PROJECT_NAME}_output/point_cloud_pruned_pose.json
        echo "  → point_cloud_pruned_pose.json"
        
        cp data/${PROJECT_NAME}_cf3/autoencoder.pth \
           data/${PROJECT_NAME}_output/autoencoder.pth
        echo "  → autoencoder.pth"
        
        cp data/${PROJECT_NAME}_cf3/feature_field.ply \
           data/${PROJECT_NAME}_output/feature_field.ply
        echo "  → feature_field.ply"
        
        echo "✓ 4개 파일 저장 완료"
        
        # 중간 파일 삭제
        echo ""
        echo "🗑️  중간 파일 삭제 중..."
        rm -rf data/${PROJECT_NAME}/
        echo "  ✓ data/${PROJECT_NAME}/ 삭제"
        
        rm -rf data/${PROJECT_NAME}_langsplat/
        echo "  ✓ data/${PROJECT_NAME}_langsplat/ 삭제"
        
        rm -rf data/${PROJECT_NAME}_cf3/
        echo "  ✓ data/${PROJECT_NAME}_cf3/ 삭제"
        
        echo ""
        echo "✓ 정리 완료"
        echo ""
        echo "📁 최종 출력 위치: data/${PROJECT_NAME}_output/"
        echo "  - point_cloud.ply"
        echo "  - point_cloud_pruned_pose.json"
        echo "  - autoencoder.pth"
        echo "  - feature_field.ply"
    fi
else
    echo ""
    echo "📁 출력 위치 (디버그 모드):"
    echo "  - 3DGS: data/${PROJECT_NAME}/output/"
    echo "  - LangSplat: data/${PROJECT_NAME}_langsplat/"
    echo "  - CF3: data/${PROJECT_NAME}_cf3/"
    echo ""
    echo "⚠️  디버그 모드: 중간 파일이 모두 유지됩니다"
fi

echo ""
echo "✓ 모든 단계 완료"
echo ""
