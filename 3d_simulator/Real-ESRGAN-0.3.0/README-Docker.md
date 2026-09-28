# Real-ESRGAN Docker Setup

이 디렉토리에는 Real-ESRGAN을 Docker 컨테이너에서 실행하기 위한 설정 파일들이 포함되어 있습니다.

## 파일 설명

- **Dockerfile**: Real-ESRGAN Docker 이미지를 빌드하기 위한 설정 파일
- **environment.yml**: Conda 환경 설정 파일
- **requirements-frozen.txt**: 정확한 버전이 고정된 Python 패키지 목록
- **docker-compose.yml**: Docker Compose를 사용한 간편한 실행 설정
- **.dockerignore**: Docker 빌드 시 제외할 파일 목록

## 사용 방법

### 1. Docker 이미지 빌드

```bash
docker build -t realesrgan:latest .
```

### 2. Docker Compose로 실행 (권장)

```bash
# 이미지 빌드 및 실행
docker-compose up

# 백그라운드에서 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 3. Docker 명령어로 직접 실행

```bash
# 기본 실행
docker run --rm \
  -v $(pwd)/tesla_motion:/app/tesla_motion:ro \
  -v $(pwd)/tesla_motion_enhanced:/app/tesla_motion_enhanced \
  -v $(pwd)/weights:/app/weights \
  realesrgan:latest \
  conda run --no-capture-output -n realesrgan \
  python inference_realesrgan.py \
  -i tesla_motion \
  -o tesla_motion_enhanced \
  -s 4 \
  --fp32 \
  --tile 400

# 커스텀 입력/출력 폴더 사용
docker run --rm \
  -v /path/to/input:/app/input:ro \
  -v /path/to/output:/app/output \
  -v $(pwd)/weights:/app/weights \
  realesrgan:latest \
  conda run --no-capture-output -n realesrgan \
  python inference_realesrgan.py \
  -i input \
  -o output \
  -s 4 \
  --fp32 \
  --tile 400
```

### 4. GPU 지원 (NVIDIA Docker 필요)

GPU를 사용하려면 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)이 설치되어 있어야 합니다.

**docker-compose.yml 수정:**
```yaml
# deploy 섹션의 주석을 제거하세요
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Docker 명령어:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/tesla_motion:/app/tesla_motion:ro \
  -v $(pwd)/tesla_motion_enhanced:/app/tesla_motion_enhanced \
  -v $(pwd)/weights:/app/weights \
  realesrgan:latest \
  conda run --no-capture-output -n realesrgan \
  python inference_realesrgan.py \
  -i tesla_motion \
  -o tesla_motion_enhanced \
  -s 4 \
  --tile 400
```

## Conda 환경 재현 (Docker 없이)

Docker를 사용하지 않고 로컬 환경을 재현하려면:

```bash
# Conda 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate realesrgan

# Real-ESRGAN 패키지 설치
pip install -e .

# 실행
python inference_realesrgan.py -i tesla_motion -o tesla_motion_enhanced -s 4 --fp32 --tile 400
```

또는 pip만 사용:

```bash
# Python 3.10 가상환경 생성
python3.10 -m venv venv_realesrgan
source venv_realesrgan/bin/activate

# 패키지 설치
pip install -r requirements-frozen.txt
pip install -e .

# 실행
python inference_realesrgan.py -i tesla_motion -o tesla_motion_enhanced -s 4 --fp32 --tile 400
```

## 주의사항

- **CPU vs GPU**: 기본적으로 CPU에서 실행됩니다. GPU를 사용하려면 위의 GPU 지원 섹션을 참조하세요.
- **메모리**: 타일 크기(--tile)를 조정하여 메모리 사용량을 제어할 수 있습니다. 메모리가 부족하면 타일 크기를 줄이세요 (예: 256, 128).
- **모델**: 기본적으로 `realesr-animevideov3` 모델을 사용합니다. 다른 모델을 사용하려면 `-n` 옵션을 변경하세요.

## 환경 사양

- Python: 3.10
- PyTorch: 1.13.1
- NumPy: 1.26.4 (NumPy 2.x 미만)
- OpenCV: 4.8.0.74

이 환경은 macOS ARM64에서 테스트되었으며, Linux x86_64 및 ARM64에서도 작동합니다.
