# 🐳 Docker로 SAM + CLIP 학습하기

Docker를 사용하여 완전히 격리된 환경에서 SAM + CLIP 모델을 학습할 수 있습니다.

## 📋 사전 요구사항

### 1. Docker 설치
```bash
# Docker 설치 확인
docker --version

# 설치되지 않은 경우
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. NVIDIA Docker Runtime 설치
```bash
# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 테스트
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 빠른 시작

### 1단계: Docker 이미지 빌드
```bash
./docker-build.sh
```

이미지 빌드에는 약 10-15분 소요됩니다. 다음이 설치됩니다:
- NVIDIA CUDA 13.0 + cuDNN
- Python 3.10
- PyTorch 2.x (CUDA 12.1)
- OpenCLIP
- Segment Anything
- 기타 필수 패키지

### 2단계: 환경 테스트
```bash
./docker-test.sh
```

다음을 확인합니다:
- GPU 접근 가능 여부
- CUDA 버전
- 데이터셋 로딩
- 모델 로딩

### 3단계: 학습 시작

#### 방법 1: 스크립트 사용 (추천)
```bash
./docker-train.sh
```

#### 방법 2: 대화형 컨테이너
```bash
# 컨테이너 시작
./docker-run.sh

# 컨테이너 내부에서
python3 train.py \
    --data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --batch-size 2 \
    --epochs 50 \
    --device cuda
```

#### 방법 3: Docker Compose
```bash
# 컨테이너 시작
docker-compose up -d

# 컨테이너 접속
docker-compose exec sam-clip-training bash

# 학습 실행
python3 train.py --data-root ./VehicleSeg10K --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth

# 컨테이너 종료
docker-compose down
```

### 4단계: TensorBoard 모니터링
```bash
./docker-tensorboard.sh

# 또는 특정 로그 디렉토리 지정
./docker-tensorboard.sh ./outputs/sam_clip_20251229_174500/logs
```

브라우저에서 http://localhost:6006 접속

## 📁 볼륨 마운트

Docker 컨테이너는 다음 디렉토리를 호스트와 공유합니다:

```
호스트                           → 컨테이너
./                              → /workspace/finetuning
./outputs                       → /workspace/finetuning/outputs
./VehicleSeg10K                 → /workspace/finetuning/VehicleSeg10K
./checkpoints                   → /workspace/finetuning/checkpoints
```

학습 결과는 호스트의 `./outputs` 디렉토리에 자동 저장됩니다.

## 🔧 Docker 스크립트 설명

### `docker-build.sh`
Docker 이미지를 빌드합니다.
```bash
./docker-build.sh
```

### `docker-run.sh`
대화형 컨테이너를 시작합니다.
```bash
./docker-run.sh
```

### `docker-train.sh`
학습을 자동으로 시작합니다 (백그라운드).
```bash
./docker-train.sh
```

### `docker-test.sh`
Docker 환경을 테스트합니다.
```bash
./docker-test.sh
```

### `docker-tensorboard.sh`
TensorBoard 서버를 시작합니다.
```bash
./docker-tensorboard.sh [로그_디렉토리]
```

## 📊 유용한 Docker 명령어

### 실행 중인 컨테이너 확인
```bash
docker ps
```

### 컨테이너 접속
```bash
docker exec -it sam-clip-train bash
```

### 컨테이너 로그 확인
```bash
docker logs -f sam-clip-train
```

### 컨테이너 중지
```bash
docker stop sam-clip-train
```

### 이미지 삭제
```bash
docker rmi sam-clip-finetuning:latest
```

### GPU 사용량 모니터링
```bash
# 호스트에서
watch -n 1 nvidia-smi

# 컨테이너 내부에서
docker exec sam-clip-train nvidia-smi
```

## 🎯 커스텀 설정

### GPU 선택
`docker-compose.yml` 또는 `docker-run.sh`에서 수정:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1  # GPU 0, 1 사용
  - CUDA_VISIBLE_DEVICES=0,1
```

### 메모리 제한
```yaml
deploy:
  resources:
    limits:
      memory: 32G
```

### 포트 변경 (TensorBoard)
```yaml
ports:
  - "6007:6006"  # 호스트:컨테이너
```

## 🐛 문제 해결

### "docker: Error response from daemon: could not select device driver"
NVIDIA Docker runtime이 설치되지 않았습니다.
```bash
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### "nvidia-smi: command not found" (컨테이너 내부)
GPU가 마운트되지 않았습니다.
```bash
# --gpus all 또는 --runtime=nvidia 옵션 확인
docker run --gpus all ...
```

### "CUDA out of memory"
배치 사이즈를 줄이세요:
```bash
python3 train.py --batch-size 1 ...
```

### 컨테이너가 즉시 종료됨
로그 확인:
```bash
docker logs sam-clip-train
```

### X11 forwarding 오류 (GUI)
```bash
xhost +local:docker
```

## 💡 팁

### 1. 개발 모드
코드를 수정하면서 작업할 때:
```bash
# 대화형 모드로 시작
./docker-run.sh

# 컨테이너 내부에서 자유롭게 작업
vim train.py
python3 train.py ...
```

### 2. Jupyter Notebook 사용
```bash
docker run -it --rm --gpus all \
    -p 8888:8888 \
    -v $(pwd):/workspace/finetuning \
    -w /workspace/finetuning \
    sam-clip-finetuning:latest \
    jupyter notebook --ip=0.0.0.0 --allow-root
```

### 3. 여러 실험 병렬 실행
```bash
# GPU 0에서 실험 1
docker run -d --name exp1 --gpus '"device=0"' ... python3 train.py --save-dir ./outputs/exp1

# GPU 1에서 실험 2
docker run -d --name exp2 --gpus '"device=1"' ... python3 train.py --save-dir ./outputs/exp2
```

### 4. 컨테이너 내부 파일 복사
```bash
# 컨테이너 → 호스트
docker cp sam-clip-train:/workspace/finetuning/outputs/model.pth ./

# 호스트 → 컨테이너
docker cp ./data.zip sam-clip-train:/workspace/finetuning/
```

## 📦 이미지 공유

### 이미지 저장
```bash
docker save sam-clip-finetuning:latest | gzip > sam-clip-finetuning.tar.gz
```

### 이미지 로드
```bash
docker load < sam-clip-finetuning.tar.gz
```

### Docker Hub에 푸시
```bash
docker tag sam-clip-finetuning:latest yourusername/sam-clip-finetuning:latest
docker push yourusername/sam-clip-finetuning:latest
```

## 🔒 보안

컨테이너는 다음 권한을 사용합니다:
- `--ipc=host`: 멀티프로세싱 지원
- `--network=host`: 호스트 네트워크 직접 사용
- `--security-opt seccomp=unconfined`: 일부 시스템 콜 제한 해제

프로덕션 환경에서는 더 엄격한 보안 정책을 고려하세요.

## 📚 참고 자료

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Documentation](https://docs.docker.com/)
- [CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda)

---

**Last Updated**: 2025-12-29
