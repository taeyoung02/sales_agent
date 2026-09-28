# Context-Aware Blur Cropping Strategy

## 문제점: 기존 Tight Bbox Cropping의 한계

### 1. **컨텍스트 손실 (Loss of Context)**
- CLIP은 "문 손잡이"와 "금속 원통"을 구별하지 못함
- 주변 컨텍스트 없이는 부품의 정체성 파악 불가
- 예: 휠만 보면 "원형 물체", 펜더와 함께 보면 "차량 휠"

### 2. **배경 노이즈 (Background Noise)**
- 랜덤한 배경 요소가 CLIP의 주의를 분산
- Tight crop은 배경을 완전히 제거하지만 컨텍스트도 제거
- 예: 창문만 crop하면 주변 차체 정보 손실

## 해결책: Context-Aware Blur Cropping

### 전략 1: Bbox 확장 (Context Preservation)
```python
expand_ratio = 0.3  # 30% 확장
pad_w = int(w * expand_ratio)
pad_h = int(h * expand_ratio)

# Bounding box를 30% 확장하여 주변 컨텍스트 포함
x1 = max(0, x_min - pad_w)
y1 = max(0, y_min - pad_h)
x2 = min(w_img, x_max + pad_w + 1)
y2 = min(h_img, y_max + pad_h + 1)
```

**효과:**
- 부품 주변의 맥락 정보 포함 (예: 휠 + 펜더, 문 + 차체)
- CLIP이 부품의 위치와 역할 파악 가능
- "어디에 부착된 부품인가?"라는 정보 보존

### 전략 2: 배경 블러링 (Focus Enhancement)
```python
# 전체 crop에 Gaussian blur 적용
blurred_img = cv2.GaussianBlur(cropped_img, (51, 51), 0)

# 마스크 영역만 선명하게, 배경은 흐리게
mask_3ch = np.stack([cropped_mask] * 3, axis=-1).astype(np.float32)
final_img = (cropped_img * mask_3ch + 
             blurred_img * (1.0 - mask_3ch))
```

**효과:**
- 배경은 흐리게 처리하여 CLIP의 주의를 마스크 영역에 집중
- 컨텍스트는 보존하되, 노이즈는 억제
- "주변 정보는 있지만 덜 중요하다"는 시각적 신호

## 구현 코드

### Dataset에서 사용 (dataset.py)
```python
# VehicleSegDataset 클래스에 추가
@staticmethod
def get_context_aware_crop(
    mask: np.ndarray,
    image: np.ndarray,
    expand_ratio: float = 0.3,
    blur_strength: Tuple[int, int] = (51, 51)
) -> np.ndarray:
    # 구현 내용은 dataset.py 참조
    ...
```

### Model에서 사용 (model.py)
```python
# SAMCLIPModel.extract_mask_features()에 통합
def extract_mask_features(
    self, 
    image: torch.Tensor, 
    mask: torch.Tensor,
    use_context_blur: bool = True,
    expand_ratio: float = 0.3,
    blur_strength: Tuple[int, int] = (51, 51)
) -> torch.Tensor:
    if use_context_blur:
        cropped_np = self._get_context_aware_crop(
            mask_np, image_np, expand_ratio, blur_strength
        )
    else:
        # Fallback to tight bbox cropping
        ...
```

### Forward Pass에 적용
```python
# model.forward()에서 자동 적용
logits = model(
    image_tensor, 
    mask_tensors,
    use_context_blur=True,  # 기본값 True
    expand_ratio=0.3,        # 30% 확장
    blur_strength=(51, 51)   # 강한 블러
)
```

## 하이퍼파라미터 튜닝 가이드

### expand_ratio (권장: 0.2 ~ 0.4)
- **0.2**: 최소 컨텍스트, 노이즈 최소화
- **0.3**: 균형잡힌 설정 (기본값)
- **0.4**: 최대 컨텍스트, 배경 정보 많음

### blur_strength (권장: (31, 31) ~ (71, 71))
- **(31, 31)**: 약한 블러, 배경 디테일 보존
- **(51, 51)**: 중간 블러 (기본값)
- **(71, 71)**: 강한 블러, 배경 완전 흐림
- **주의**: 커널 크기는 반드시 홀수여야 함

## preprocess.py와의 일관성

이 구현은 `preprocess.py`의 `get_context_aware_crop()` 함수와 동일한 로직을 사용합니다:

**preprocess.py (Feature Extraction)**
```python
seg_img = get_context_aware_crop(
    mask, 
    image, 
    expand_ratio=0.3,
    blur_strength=(51, 51)
)
```

**model.py (Fine-tuning)**
```python
cropped_np = self._get_context_aware_crop(
    mask_np, image_np, 
    expand_ratio=0.3,
    blur_strength=(51, 51)
)
```

이를 통해 학습 시와 추론 시 동일한 전처리 파이프라인을 사용하여 **train-test consistency**를 보장합니다.

## 기대 효과

### 1. **Fine-grained Recognition 개선**
- 작은 부품 (엠블럼, 손잡이, 휠 스포크)의 분류 정확도 향상
- 컨텍스트 정보로 모호성 해소

### 2. **CLIP Feature Quality 향상**
- 배경 노이즈 감소로 더 깨끗한 feature 추출
- 마스크 영역에 집중된 attention

### 3. **Robustness 증가**
- 다양한 촬영 각도와 거리에서 일관된 성능
- 배경 변화에 강건함

## 비활성화 방법

필요시 context-aware blur를 비활성화하고 기존 tight bbox cropping으로 되돌릴 수 있습니다:

```python
# Forward pass에서
logits = model(
    image_tensor, 
    mask_tensors,
    use_context_blur=False  # 비활성화
)

# 또는 extract_mask_features 직접 호출 시
features = model.extract_mask_features(
    image, mask,
    use_context_blur=False
)
```

## 참고 자료

- Original implementation: `preprocess.py` (line 179-267)
- Dataset integration: `dataset.py` (line 142-221)
- Model integration: `model.py` (line 71-223)
- Training usage: `train.py` (forward pass에서 자동 적용)
