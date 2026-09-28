# SAM + CLIP Fine-tuning Project Structure

## 📁 프로젝트 구조

```
finetuning/
├── 📄 README.md                    # 프로젝트 문서
├── 📄 requirements.txt             # Python 패키지 의존성
├── 🔧 setup.sh                     # 환경 설정 스크립트
├── 🚀 run_training.sh              # 학습 실행 스크립트
├── 🧪 test_setup.py                # 환경 테스트 스크립트
│
├── 📊 dataset.py                   # VehicleSegDataset 클래스
├── 🤖 model.py                     # SAMCLIPModel 클래스
├── 🏋️ train.py                     # 학습 스크립트
├── 🔮 inference.py                 # 추론 스크립트
│
├── 📦 checkpoints/                 # SAM 체크포인트
│   └── sam_vit_b_01ec64.pth
│
└── 📂 VehicleSeg10K/               # 데이터셋
    ├── images/
    │   ├── train/
    │   └── valid/
    └── annotations/
        ├── train/
        └── valid/
```

## 🎯 핵심 기능

### 1. Dataset (`dataset.py`)
- **VehicleSegDataset**: VehicleSeg10K 데이터셋 로더
- Polygon annotation을 binary mask로 변환
- SAM 마스크와 GT 마스크 IoU 매칭
- 13개 차량 부품 라벨 + "Others"

### 2. Model (`model.py`)
- **SAMCLIPModel**: SAM + CLIP 통합 모델
- SAM (frozen): 고품질 마스크 생성
- CLIP Vision Encoder (trainable): 마스크 영역 특징 추출
- Classification Head (trainable): 14-class 분류

### 3. Training (`train.py`)
- GPU 가속 학습 (CUDA 13.0)
- AdamW optimizer + Cosine Annealing LR
- TensorBoard 로깅
- Checkpoint 저장 (주기적 + best model)

### 4. Inference (`inference.py`)
- 단일 이미지 추론
- 결과 시각화 (matplotlib)
- JSON 결과 저장

## 🚀 빠른 시작

### 1단계: 환경 설정
```bash
./setup.sh
```

### 2단계: 환경 테스트
```bash
python test_setup.py
```

### 3단계: 학습 시작
```bash
./run_training.sh
```

또는 수동으로:
```bash
python train.py \
    --data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --batch-size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --device cuda \
    --gpu-id 0
```

### 4단계: 추론
```bash
python inference.py \
    --checkpoint ./outputs/sam_clip_XXXXXX/best_model.pth \
    --image ./test_image.jpg \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --output-dir ./outputs/inference \
    --device cuda \
    --show
```

## 📊 데이터셋 정보

### 클래스 (13개 + Others)
1. **Wheel** (20,976개) - 바퀴
2. **Foreground** (8,601개) - 전경
3. **Left back window** (5,701개) - 좌측 뒷창
4. **Plate** (5,888개) - 번호판
5. **Front window** (4,559개) - 앞창
6. **Right back window** (4,432개) - 우측 뒷창
7. **Left front window** (3,960개) - 좌측 앞창
8. **Left front door** (3,734개) - 좌측 앞문
9. **Left back door** (3,268개) - 좌측 뒷문
10. **Right front window** (3,057개) - 우측 앞창
11. **Right front door** (2,862개) - 우측 앞문
12. **Back window** (2,916개) - 뒷창
13. **Right back door** (2,555개) - 우측 뒷문
14. **Others** - 매칭되지 않는 영역

## 🎓 학습 전략

### SAM (Frozen)
- 사전학습된 가중치 사용
- 마스크 생성만 활용
- GPU 메모리 효율적

### CLIP Vision Encoder (Trainable)
- OpenAI 사전학습 모델 기반
- 차량 부품 이미지에 맞게 fine-tuning
- Feature extraction 학습

### Classification Head (Trainable)
- 2-layer MLP
- Dropout 0.3
- 14-class classification

## 🔧 주요 하이퍼파라미터

```python
# Model
sam_model_type = 'vit_b'
clip_model_name = 'ViT-B-32'
num_classes = 14

# Training
batch_size = 2-4 (GPU 메모리에 따라)
learning_rate = 1e-4
weight_decay = 1e-4
epochs = 50
optimizer = AdamW
scheduler = CosineAnnealingLR

# Data
iou_threshold = 0.5 (GT 매칭)
```

## 📈 모니터링

### TensorBoard
```bash
tensorboard --logdir ./outputs/sam_clip_XXXXXX/logs
```

추적 메트릭:
- Train/Loss
- Train/Accuracy
- Valid/Loss
- Valid/Accuracy
- Train/LearningRate

## 💾 출력 구조

```
outputs/
└── sam_clip_20251229_174500/
    ├── logs/                        # TensorBoard 로그
    │   └── events.out.tfevents.*
    ├── checkpoint_epoch_5.pth       # 주기적 체크포인트
    ├── checkpoint_epoch_10.pth
    ├── ...
    └── best_model.pth              # Best validation accuracy
```

## 🐛 문제 해결

### CUDA Out of Memory
- `--batch-size` 줄이기 (2 또는 1)
- SAM mask 생성 시 `points_per_side` 줄이기

### Slow Training
- `--num-workers` 증가 (CPU 코어 수만큼)
- Mixed precision training 추가 고려

### Low Accuracy
- `--epochs` 증가
- Learning rate 조정
- Data augmentation 추가

## 📚 참고 자료

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [PyTorch](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 🙏 Credits

- Meta AI Research - SAM
- OpenAI - CLIP
- VehicleSeg10K Dataset

---

**GPU**: NVIDIA GB10 (CUDA 13.0)  
**Last Updated**: 2025-12-29
