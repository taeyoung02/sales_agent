# Mask-based Voting Culling for SuGaR - Implementation Guide

## 개요 (Overview)

이 구현은 SuGaR의 Mesh Extraction(Poisson Reconstruction) 단계 직전에 배경 Gaussian을 제거하는 **"Mask-based Voting Culling"** 알고리즘을 추가합니다.

알고리즘의 핵심 아이디어:
- 모든 Gaussian을 각 카메라 뷰에 투영(Projection)
- 각 뷰의 Binary Mask(0=배경, 1=객체)를 확인
- 다수의 카메라에서 마스크 바깥(배경)에 위치한 Gaussian을 삭제

## 구현 파일 (Implementation Files)

### 1. `sugar_utils/mask_voting_culling.py`
핵심 알고리즘 구현:
- `load_binary_masks()`: 마스크 이미지 로딩 및 전처리
- `cull_gaussians_by_mask_voting()`: 투표 기반 Gaussian 제거

### 2. `extract_mesh.py` (수정됨)
새로운 명령줄 인자 추가:
- `--mask_dir`: 마스크 이미지 디렉토리 경로
- `--enable_mask_culling`: 마스크 컬링 활성화 여부
- `--mask_voting_threshold`: 배경 투표 임계값 (0.0~1.0)
- `--mask_dilation_kernel`: 마스크 확장 커널 크기
- `--debug_culling`: 디버그 시각화 활성화

### 3. `sugar_extractors/coarse_mesh.py` (수정됨)
통합 위치:
- Low opacity pruning 직후
- Triangle soup 생성 직전
- Poisson reconstruction 전

## 사용 방법 (Usage)

### 기본 사용법

```bash
python extract_mesh.py \
    -s /path/to/scene/data \
    -c /path/to/gaussian_splatting/checkpoint \
    -m /path/to/coarse_sugar_model.ckpt \
    --mask_dir /path/to/masks \
    --enable_mask_culling True \
    --mask_voting_threshold 0.75 \
    --mask_dilation_kernel 7
```

### 파라미터 설명

#### `--mask_dir` (필수)
Binary mask 이미지가 있는 디렉토리 경로.
- 마스크 파일명: `{image_name}.png` 또는 `{image_name}.jpg`
- 마스크 값: 0=배경, 255=객체 (grayscale)
- 예: `data/volvo/masks/`

#### `--mask_voting_threshold` (기본값: 0.75)
배경으로 판단하는 투표 비율 임계값:
- `1.0`: 모든 카메라에서 배경으로 보여야만 삭제 (가장 안전)
- `0.8`: 80% 이상의 카메라에서 배경이면 삭제 (권장)
- `0.5`: 절반 이상의 카메라에서 배경이면 삭제 (공격적)

**권장값: 0.7 ~ 0.8**

#### `--mask_dilation_kernel` (기본값: 7)
마스크를 확장하는 커널 크기 (픽셀):
- `0`: 확장 없음
- `5-10`: 권장 범위
- 큰 값: 객체 경계면의 Gaussian 보호 (안전)
- 작은 값: 더 깔끔한 배경 제거

**권장값: 5~10**

#### `--debug_culling` (기본값: False)
디버그 시각화 활성화:
- `True`: 투영 결과, 통계, 시각화 이미지 저장
- 저장 위치: `{mesh_output_dir}/culling_debug/`

### 예제 명령어

#### 1. Volvo 차량 (권장 설정)
```bash
python extract_mesh.py \
    -s data/volvo \
    -c output/volvo/point_cloud/iteration_7000 \
    -m output/volvo/coarse_sugar_7k.ckpt \
    -i 7000 \
    --mask_dir data/volvo/masks \
    --enable_mask_culling True \
    --mask_voting_threshold 0.75 \
    --mask_dilation_kernel 7 \
    --debug_culling True
```

#### 2. 보수적 설정 (객체 손실 최소화)
```bash
python extract_mesh.py \
    -s data/volvo \
    -c output/volvo/point_cloud/iteration_7000 \
    -m output/volvo/coarse_sugar_7k.ckpt \
    --mask_dir data/volvo/masks \
    --mask_voting_threshold 0.9 \
    --mask_dilation_kernel 10
```

#### 3. 공격적 설정 (깔끔한 배경 제거)
```bash
python extract_mesh.py \
    -s data/volvo \
    -c output/volvo/point_cloud/iteration_7000 \
    -m output/volvo/coarse_sugar_7k.ckpt \
    --mask_dir data/volvo/masks \
    --mask_voting_threshold 0.6 \
    --mask_dilation_kernel 5
```

## 마스크 준비 (Mask Preparation)

### rembg를 사용한 마스크 생성

```bash
# 이미 프로젝트에 있는 스크립트 사용
python generate_rembg_masks.py \
    --input_dir data/volvo/images \
    --output_dir data/volvo/masks
```

### 마스크 요구사항

1. **파일 형식**: PNG 또는 JPG (grayscale)
2. **파일명**: 카메라 이미지와 동일한 이름
3. **값 범위**:
   - 0 (검정): 배경
   - 255 (흰색): 객체/전경
4. **해상도**: 카메라 이미지와 동일 (자동 리사이징됨)

### 마스크 품질 확인

```python
import cv2
import numpy as np

# 마스크 로딩 및 확인
mask = cv2.imread('data/volvo/masks/frame_001.png', cv2.IMREAD_GRAYSCALE)
print(f"마스크 크기: {mask.shape}")
print(f"고유 값: {np.unique(mask)}")
print(f"배경 비율: {(mask == 0).sum() / mask.size * 100:.1f}%")
print(f"객체 비율: {(mask > 0).sum() / mask.size * 100:.1f}%")

# 시각화
import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')
plt.show()
```

## 알고리즘 동작 원리 (How It Works)

### 1. 초기화
```
background_votes[N] = 0  # N = Gaussian 개수
visible_counts[N] = 0
```

### 2. 각 카메라 뷰에 대해 반복

```python
for camera in cameras:
    # A. 3D 점을 2D 이미지 평면에 투영
    points_2d = project_to_camera(points_3d, camera)
    
    # B. 화면 내부에 있는 점만 선택
    valid_points = filter_in_frame(points_2d)
    
    # C. 마스크 샘플링
    mask_values = sample_mask(mask, valid_points)
    
    # D. 투표
    for point in valid_points:
        if mask_values[point] == 0:  # 배경
            background_votes[point] += 1
        visible_counts[point] += 1
```

### 3. 결정

```python
ratios = background_votes / visible_counts
to_delete = ratios > threshold

# SuGaR 모델에서 삭제
sugar._points = sugar._points[~to_delete]
sugar._scales = sugar._scales[~to_delete]
# ... 기타 속성들
```

## 디버그 출력 (Debug Output)

`--debug_culling True` 사용 시 생성되는 파일들:

### 1. `culling_debug/culling_stats.json`
```json
{
  "initial_gaussians": 250000,
  "deleted_gaussians": 125000,
  "kept_gaussians": 125000,
  "deletion_ratio": 0.5,
  "threshold": 0.75,
  "processed_views": 50
}
```

### 2. `culling_debug/projection_*.jpg`
- 처음 5개 카메라 뷰의 투영 시각화
- 빨간 점: 배경으로 판단된 Gaussian
- 초록 점: 객체로 판단된 Gaussian

### 3. 콘솔 출력 예시
```
================================================================================
Starting Mask-based Voting Culling
================================================================================
Initial number of Gaussians: 250,000
Voting threshold: 0.75
Number of cameras with masks: 50

Projecting Gaussians to camera views...
  Processed 10 views...
  Processed 20 views...
  ...
Completed projection for 50 views

Voting Results:
  Points to delete: 125,000 (50.0%)
  Points to keep:   125,000 (50.0%)

Background Voting Ratio Distribution:
    0% quantile: 0.000
   25% quantile: 0.400
   50% quantile: 0.800
   75% quantile: 0.950
   90% quantile: 1.000
   95% quantile: 1.000
  100% quantile: 1.000

✓ Culling complete! Final Gaussian count: 125,000
================================================================================
```

## Best Practices

### 1. 마스크 Dilation (확장)
- **중요!** rembg 마스크는 경계가 타이트함
- 3DGS는 객체 경계를 약간 넘어서 표현
- **권장**: `mask_dilation_kernel=7` 또는 `10`

### 2. Threshold 조정
- **테스트 방법**:
  1. 먼저 `threshold=0.9`로 시작 (안전)
  2. 배경이 남아있으면 점차 낮춤
  3. 객체가 손상되면 다시 높임
- **최적값**: 대부분 `0.7~0.8`

### 3. 메모리 관리
- 마스크를 미리 모두 로딩하지 않음
- Loop 내에서 필요시 로딩
- GPU 메모리 부족 시 `visible_counts` 체크 주기 조정

### 4. 디버깅 워크플로우
```bash
# 1. 디버그 모드로 실행
python extract_mesh.py ... --debug_culling True

# 2. 결과 확인
ls output/coarse_mesh/volvo/culling_debug/
cat output/coarse_mesh/volvo/culling_debug/culling_stats.json

# 3. 투영 시각화 확인
open output/coarse_mesh/volvo/culling_debug/projection_*.jpg

# 4. 파라미터 조정 후 재실행
```

## 통합 파이프라인 예제

### 전체 워크플로우

```bash
#!/bin/bash

SCENE_DIR="data/volvo"
OUTPUT_DIR="output/volvo"

# Step 1: Generate masks
echo "Generating masks..."
python generate_rembg_masks.py \
    --input_dir ${SCENE_DIR}/images \
    --output_dir ${SCENE_DIR}/masks

# Step 2: Train vanilla 3DGS (7k iterations)
echo "Training vanilla 3DGS..."
python gaussian_splatting/train.py \
    -s ${SCENE_DIR} \
    -m ${OUTPUT_DIR} \
    --iterations 7000

# Step 3: Train coarse SuGaR
echo "Training coarse SuGaR..."
python train.py \
    -s ${SCENE_DIR} \
    -c ${OUTPUT_DIR}/point_cloud/iteration_7000 \
    -r "density" \
    --iterations 7000

# Step 4: Extract mesh with mask culling
echo "Extracting mesh with background removal..."
python extract_mesh.py \
    -s ${SCENE_DIR} \
    -c ${OUTPUT_DIR}/point_cloud/iteration_7000 \
    -m ${OUTPUT_DIR}/coarse_sugar_7k.ckpt \
    -i 7000 \
    --mask_dir ${SCENE_DIR}/masks \
    --enable_mask_culling True \
    --mask_voting_threshold 0.75 \
    --mask_dilation_kernel 7 \
    --debug_culling True

echo "Done! Check output in ${OUTPUT_DIR}/coarse_mesh/"
```

## 문제 해결 (Troubleshooting)

### 1. 마스크가 로딩되지 않음
```
Warning: Mask not found for camera: frame_001
```
**해결**:
- 파일명 확인: `{camera.image_name}.png`
- 확장자 확인: `.png`, `.jpg`, `.PNG`, `.JPG`
- 경로 확인: `--mask_dir` 옵션이 올바른지

### 2. 객체의 일부가 삭제됨
```
Points to delete: 200,000 (80.0%)
```
**해결**:
- `--mask_voting_threshold` 증가 (예: 0.9)
- `--mask_dilation_kernel` 증가 (예: 10)
- 마스크 품질 확인 (경계면이 충분히 포함되는지)

### 3. 배경이 너무 많이 남음
```
Points to delete: 10,000 (4.0%)
```
**해결**:
- `--mask_voting_threshold` 감소 (예: 0.6)
- `--mask_dilation_kernel` 감소 (예: 5)
- 마스크가 너무 큰지 확인

### 4. GPU 메모리 부족
```
CUDA out of memory
```
**해결**:
- Low opacity pruning threshold 증가 (더 많은 Gaussian 미리 제거)
- 디버그 모드 비활성화 (`--debug_culling False`)

## 성능 및 효과

### 예상 결과
- **삭제 비율**: 30~60% (배경이 많을수록 높음)
- **처리 시간**: 카메라당 약 0.1~0.5초
- **메모리 사용**: 마스크당 약 5~10MB

### Mesh 품질 향상
- ✅ 배경 노이즈 제거
- ✅ Poisson reconstruction 안정성 향상
- ✅ 파일 크기 감소
- ✅ 렌더링 속도 향상

## 참고 자료 (References)

- SuGaR Paper: https://arxiv.org/abs/2311.12775
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Rembg: https://github.com/danielgatis/rembg

## 라이선스 (License)

이 구현은 SuGaR 프로젝트의 라이선스를 따릅니다.
