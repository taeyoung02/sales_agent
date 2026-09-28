# Debugging & Visualization Guide

## 목적

`debugging.py`는 학습된 SAM + CLIP 모델을 분석하고 다음을 시각화합니다:

1. **SAM 마스크 생성 과정**: 어떤 영역을 감지했는지, 마스크 품질은 어떤지
2. **Context-Aware Blur Cropping**: 전처리 전/후 비교
3. **CLIP 분류 결과**: 각 마스크에 어떤 라벨을 할당했는지, 신뢰도는 얼마인지
4. **Ground Truth 비교**: 실제 정답과 얼마나 일치하는지

## 사용법

### 1. 빠른 시작 (추천)

검증 데이터셋에서 랜덤하게 5개 샘플을 선택하여 자동 분석:

```bash
./docker-debug.sh
```

**출력:**
- `debug_output/` 디렉토리에 각 이미지당 4개의 시각화 파일 생성

### 2. 특정 이미지 분석

```bash
# Docker 컨테이너 진입
./docker-run.sh

# 컨테이너 내부에서
python3 debugging.py \
    --image ./VehicleSeg10K/images/valid/000044576.jpg \
    --checkpoint ./outputs/best_model.pth \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --data-root ./VehicleSeg10K \
    --output-dir my_debug_output
```

### 3. 커스텀 설정

```bash
python3 debugging.py \
    --image /path/to/custom/image.jpg \
    --checkpoint ./outputs/sam_clip_20251230_120000/best_model.pth \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --output-dir custom_debug \
    --device cuda \
    --show  # 저장 + 화면에 표시
```

## 매개변수

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--image` | 분석할 이미지 경로 (필수) | - |
| `--checkpoint` | 학습된 모델 체크포인트 | `outputs/best_model.pth` |
| `--sam-checkpoint` | SAM 체크포인트 | `./checkpoints/sam_vit_b_01ec64.pth` |
| `--data-root` | VehicleSeg10K 데이터셋 루트 (GT 비교용) | `None` |
| `--output-dir` | 시각화 저장 디렉토리 | `debug_output` |
| `--show` | 화면에 plot 표시 (플래그) | False (저장만) |
| `--device` | 연산 장치 | `cuda` |

## 출력 파일 상세

### 1. `*_1_sam_masks.png` - SAM 마스크 생성 분석

**6개 서브플롯:**

1. **Original Image**: 입력 이미지
2. **All Masks Overlay**: 모든 마스크를 색상별로 오버레이
3. **Mask Boundaries**: 마스크 경계선만 표시
4. **Stability Scores**: SAM의 stability score 히트맵
5. **Predicted IoU**: SAM의 predicted IoU 히트맵
6. **Mask Size Distribution**: 마스크 크기 분포 히스토그램

**활용:**
- SAM이 얼마나 많은 마스크를 생성했는지 확인
- 작은 부품(엠블럼, 손잡이)도 잘 감지했는지 확인
- 마스크 품질 평가 (stability, IoU)

### 2. `*_2_context_aware_cropping.png` - 전처리 비교

**각 마스크당 6개 컬럼:**

1. **Original + BBox**: 원본 이미지 + tight bbox (빨강) + expanded bbox (노랑)
2. **Mask Region**: 마스크 영역만 표시
3. **Original Method**: Tight crop + 검은 배경 (기존 방식)
4. **Expanded Crop**: 30% 확장 crop (blur 없음)
5. **Context-Aware Blur**: 30% 확장 + 배경 blur (최종 방식) ⭐
6. **Difference Heatmap**: 기존 방식과의 차이

**활용:**
- Context-aware blur가 어떻게 작동하는지 시각적 확인
- 배경이 흐려지고 마스크 영역은 선명한지 확인
- 컨텍스트 보존 효과 확인

### 3. `*_3_clip_predictions.png` - CLIP 분류 결과

**4개 서브플롯:**

1. **Original Image**: 입력 이미지
2. **Predictions Overlay**: 라벨과 신뢰도가 표시된 오버레이
   - 각 마스크에 `라벨명\n신뢰도` 형태로 텍스트 표시
   - 같은 라벨은 같은 색상으로 표시
3. **Confidence Distribution**: 마스크별 신뢰도 막대 그래프
   - 평균 신뢰도 점선으로 표시
4. **Label Distribution**: 라벨별 빈도 막대 그래프

**활용:**
- CLIP이 각 부품을 올바르게 분류했는지 확인
- 낮은 신뢰도 예측 찾기
- "Others" (배경) 클래스가 제대로 할당되었는지 확인

### 4. `*_4_gt_comparison.png` - Ground Truth 비교

**4개 서브플롯:**

1. **Ground Truth**: 실제 정답 마스크 (색상별)
2. **SAM + CLIP Predictions**: 모델 예측 마스크
3. **Matching Visualization**: 
   - 🟢 Green: 정답 (IoU≥0.5, 라벨 일치)
   - 🔴 Red: 오답 (IoU≥0.5, 라벨 불일치)
   - ⚪ Gray: GT 매칭 없음 (IoU<0.5)
4. **Confusion Matrix**: GT vs 예측 혼동 행렬

**활용:**
- 모델 성능 정량적 평가
- 자주 혼동하는 라벨 쌍 찾기
- False positives/negatives 분석

## 실전 예제

### 예제 1: 학습 후 품질 체크

학습이 완료되면 먼저 랜덤 샘플을 확인:

```bash
./docker-debug.sh
```

결과 확인:
```bash
ls -lh debug_output/

# 각 이미지당 4개 파일 확인
# 평균 신뢰도, 라벨 분포 등을 summary 출력에서 확인
```

### 예제 2: 특정 실패 케이스 분석

Validation 중 특정 이미지에서 낮은 정확도가 나왔다면:

```bash
./docker-run.sh

# 컨테이너 내부
python3 debugging.py \
    --image ./VehicleSeg10K/images/valid/problem_image.jpg \
    --checkpoint ./outputs/best_model.pth \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --data-root ./VehicleSeg10K \
    --output-dir debug_failed_cases
```

시각화 확인:
1. SAM 마스크가 부품을 잘 분리했는지 (`*_1_sam_masks.png`)
2. Context-aware blur가 도움이 되었는지 (`*_2_context_aware_cropping.png`)
3. CLIP 신뢰도가 낮은 마스크는 어떤 것인지 (`*_3_clip_predictions.png`)
4. 어떤 라벨을 헷갈렸는지 (`*_4_gt_comparison.png` confusion matrix)

### 예제 3: Background Embedding 검증

"Others" 클래스가 제대로 배경을 분류하는지 확인:

```bash
python3 debugging.py \
    --image ./VehicleSeg10K/images/valid/outdoor_scene.jpg \
    --checkpoint ./outputs/best_model.pth \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --output-dir debug_background

# 출력에서 "Others" 라벨 분포 확인
# *_3_clip_predictions.png에서 도로/아스팔트가 Others로 분류되었는지 확인
```

## 해석 가이드

### 좋은 결과의 신호:
- ✅ SAM이 각 부품을 개별 마스크로 분리 (휠, 창문, 문 등)
- ✅ Context-aware blur에서 마스크 영역은 선명, 배경은 흐림
- ✅ CLIP 평균 신뢰도 > 0.7
- ✅ "Others" 클래스가 주로 도로, 배경에 할당됨
- ✅ Confusion matrix 대각선에 값이 집중 (정확한 분류)

### 문제의 신호:
- ❌ SAM이 너무 많거나 적은 마스크 생성 (>100 or <5)
- ❌ 작은 부품(엠블럼, 손잡이)을 놓침
- ❌ CLIP 신뢰도 < 0.5 (매우 불확실한 예측)
- ❌ Wheel과 Plate 혼동 (confusion matrix에서 off-diagonal 높음)
- ❌ 차량 부품이 "Others"로 분류됨

## 트러블슈팅

### GPU 메모리 부족
```bash
# SAM 생성 마스크 수 줄이기
python3 debugging.py --image ... 
# model.py에서 generate_sam_masks() 호출 시 points_per_side 감소
```

### 시각화가 너무 작음
```python
# debugging.py 수정
# figsize 값 증가: figsize=(20, 16) → figsize=(30, 24)
```

### Ground Truth 파일 없음
- `--data-root` 옵션 제거하면 GT 비교를 건너뜀
- 4번째 시각화(`*_4_gt_comparison.png`)는 생성되지 않음

## 고급 활용

### 배치 처리

여러 이미지를 한 번에 처리:

```bash
#!/bin/bash
for img in ./VehicleSeg10K/images/valid/*.jpg; do
    python3 debugging.py \
        --image "$img" \
        --checkpoint ./outputs/best_model.pth \
        --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
        --output-dir batch_debug
done
```

### 커스텀 분석 추가

`debugging.py`를 수정하여 추가 분석:

```python
# 예: CLIP feature space 분석
def visualize_feature_space(features, labels):
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    for label in set(labels):
        mask = [l == label for l in labels]
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=label)
    plt.legend()
    plt.title('CLIP Feature Space (PCA)')
    plt.savefig('feature_space.png')
```

## 참고 자료

- SAM mask quality metrics: [SAM paper](https://arxiv.org/abs/2304.02643)
- CLIP confidence calibration: [OpenCLIP docs](https://github.com/mlfoundations/open_clip)
- Context-aware cropping strategy: [CONTEXT_AWARE_BLUR.md](CONTEXT_AWARE_BLUR.md)
- Background embedding strategy: [BACKGROUND_EMBEDDING.md](BACKGROUND_EMBEDDING.md)

---

**Tips:**
- 학습 초기에는 신뢰도가 낮고 "Others"가 많이 나올 수 있음 (정상)
- 에폭이 진행되면서 신뢰도가 올라가고 라벨 분포가 GT와 유사해짐
- Confusion matrix를 보고 어떤 라벨 쌍이 어려운지 파악 → 데이터 증강 방향 결정
