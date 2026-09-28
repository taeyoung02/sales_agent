# SNUKDT 11기 코오롱 모빌리티그룹 캡스톤 프로젝트: Multi container Docker Guide
해당 프로젝트에서 구현된 멀티 컨테이너 구조와 관련된 설정 방법에 대해 안내합니다.

## 기본 환경 구성
- Docker 버전: 28.5.1 이상
- Docker Compose 버전: v2.40.0 이상 (Compose V2 형식 사용)
- NVIDIA Container Toolkit (필수)
- CUDA 13.0 지원

## 1. Docker Compose 파일 구조
프로젝트 루트 디렉토리에는 `docker-compose.yml` 파일이 위치해 있습니다. 이 파일은 4개의 주요 서비스 컨테이너를 정의합니다:

### GPU 설정 (Compose V2 형식)
모든 서비스는 다음과 같은 GPU 설정을 사용합니다:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**주의**: Docker Compose V2에서는 `runtime: nvidia` 대신 `deploy.resources` 형식을 사용합니다.

## 2. 4개 컨테이너 서비스 구조

### 2.1 Real-ESRGAN (이미지 전처리)
**목적**: 저해상도 이미지 업스케일링 및 품질 향상

```yaml
realesrgan:
  image: realesrgan-preprocessing:latest
  container_name: realesrgan_preprocessor
  shm_size: '8gb'
  volumes:
    - ./data:/app/data
    - ./Real-ESRGAN-0.3.0/weights:/app/weights
```

**주요 기능**:
- RealESRGAN_x4plus 모델 사용
- 이미지 해상도 향상 (최대 4배)
- GPU 가속 처리

**실행 예시**:
```bash
docker compose run --rm realesrgan
```

### 2.2 Rembg (배경 제거)
**목적**: 자동 배경 제거 및 마스크 생성

```yaml
rembg:
  image: rembg-cuda:latest
  container_name: rembg_mask_generator
  shm_size: '8gb'
  volumes:
    - ./data:/app/data
```

**주요 기능**:
- U2Net 모델 기반 배경 제거
- PyTorch 2.9 + CUDA 13.0
- Alpha matting 지원

**실행 예시**:
```bash
docker compose run --rm rembg python /app/generate_rembg_masks.py \
  --images_dir /app/data/my_project/images \
  --output_dir /app/data/my_project/masks
```

### 2.3 Pipeline (메인 3D 재구성)
**목적**: COLMAP, 3D Gaussian Splatting, LangSplat, CF3 실행

```yaml
pipeline:
  image: 3d-reconstruction-pipeline:latest
  container_name: 3d_recon_pipeline
  shm_size: '64gb'
  stdin_open: true
  tty: true
  volumes:
    - ./data:/workspace/data
    - ./checkpoints:/workspace/data/checkpoints
    - ./custom_models:/workspace/custom_models
  depends_on:
    - realesrgan
```

**주요 기능**:
- COLMAP 카메라 포즈 추정
- 3D Gaussian Splatting 학습
- LangSplat 특징 추출
- CF3 3D 세그멘테이션

**실행 예시**:
```bash
# Interactive 모드로 실행
docker compose run --rm pipeline bash

# 또는 스크립트 직접 실행
docker compose run --rm pipeline python3 /workspace/scripts/run_pipeline.py
```

### 2.4 SuGaR (메쉬 생성)
**목적**: Surface-Aligned Gaussian Splatting 기반 3D 메쉬 생성

```yaml
sugar:
  image: sugar-meshing:latest
  container_name: sugar_meshing
  shm_size: '64gb'
  stdin_open: true
  tty: true
  volumes:
    - ./data:/workspace/data
  environment:
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**주요 기능**:
- Vanilla 3DGS 학습
- Coarse/Refined SuGaR 생성
- UV 텍스처 메쉬 추출
- Python 3.10 + Open3D 호환

**실행 예시**:
```bash
docker compose run --rm sugar bash
```

## 3. 네트워크 구조
모든 컨테이너는 `pipeline_net`이라는 브리지 네트워크로 연결되어 서로 통신할 수 있습니다:

```yaml
networks:
  pipeline_net:
    driver: bridge
```

컨테이너 간 통신 예시:
```bash
# pipeline 컨테이너에서 rembg 컨테이너 접근
ping rembg_mask_generator
```

## 4. 전체 이미지 빌드

### 4.1 모든 이미지 한 번에 빌드
```bash
docker compose build
```

### 4.2 특정 서비스만 빌드
```bash
# Real-ESRGAN만 빌드
docker compose build realesrgan

# Pipeline과 SuGaR 빌드
docker compose build pipeline sugar
```

### 4.3 캐시 없이 재빌드
```bash
docker compose build --no-cache
```

### 4.4 병렬 빌드 (빠른 빌드)
```bash
docker compose build --parallel
```

## 5. 컨테이너 실행 방법

### 5.1 백그라운드 실행
```bash
# 모든 서비스 시작
docker compose up -d

# 특정 서비스만 시작
docker compose up -d pipeline sugar
```

### 5.2 일회성 실행 (작업 후 자동 삭제)
```bash
docker compose run --rm [service_name] [command]

# 예시
docker compose run --rm pipeline bash
docker compose run --rm sugar python3 /workspace/SuGaR/train.py
```

### 5.3 Interactive 모드
```bash
# Pipeline 컨테이너 접속
docker compose run --rm pipeline bash

# SuGaR 컨테이너 접속
docker compose run --rm sugar bash
```

### 5.4 서비스 중지 및 삭제
```bash
# 실행 중인 컨테이너 중지
docker compose stop

# 컨테이너 삭제
docker compose down

# 볼륨까지 삭제 (주의: 데이터 손실)
docker compose down -v
```

## 6. 데이터 볼륨 구조

모든 컨테이너는 `./data` 디렉토리를 공유합니다:

```
./data/
├── my_project/
│   ├── input/              # 원본 이미지 (Real-ESRGAN 입력)
│   ├── enhanced/           # 업스케일된 이미지 (Real-ESRGAN 출력)
│   ├── images/             # COLMAP용 이미지
│   ├── masks/              # Rembg 마스크 (선택적)
│   ├── sparse/             # COLMAP 카메라 포즈
│   ├── output_vanilla/     # Vanilla 3DGS
│   ├── output_coarse/      # Coarse SuGaR
│   ├── output_refined/     # Refined SuGaR
│   └── mesh_textured/      # 최종 UV 텍스처 메쉬
```

## 7. 실전 워크플로우

### 7.1 완전 자동화 파이프라인
```bash
# 1. 이미지 업스케일링 (선택적)
docker compose run --rm realesrgan

# 2. 배경 제거 (선택적)
docker compose run --rm rembg python /app/generate_rembg_masks.py \
  --images_dir /app/data/my_project/images \
  --output_dir /app/data/my_project/masks

# 3. COLMAP + 3DGS (Pipeline 스크립트 사용)
docker compose run --rm pipeline bash -c "cd /workspace && ./run_pipeline.sh"

# 4. SuGaR 메쉬 생성 (스크립트 사용)
docker compose run --rm sugar bash -c "cd /workspace && ./sugar_3dgs_script.sh"
```

### 7.2 단계별 수동 실행
```bash
# Pipeline 컨테이너 접속
docker compose run --rm pipeline bash

# 내부에서 작업
> cd /workspace
> python3 scripts/colmap_pipeline.py --project my_project
> python3 scripts/train_3dgs.py --project my_project
> exit

# SuGaR 컨테이너 접속
docker compose run --rm sugar bash

# 내부에서 작업
> cd /workspace/SuGaR
> python gaussian_splatting/train.py -s /workspace/data/my_project
> exit
```

## 8. GPU 메모리 관리

### 8.1 Shared Memory 설정
각 컨테이너는 다음과 같이 설정되어 있습니다:
- Real-ESRGAN, Rembg: 8GB
- Pipeline, SuGaR: 64GB

부족한 경우 `docker-compose.yml`에서 수정:
```yaml
shm_size: '128gb'  # 증가
```

### 8.2 GPU 메모리 확인
```bash
# 호스트에서
nvidia-smi

# 컨테이너 내에서
docker compose run --rm pipeline nvidia-smi
```

### 8.3 메모리 부족 문제 해결
```bash
# PyTorch 메모리 정리
docker compose run --rm pipeline python3 -c "import torch; torch.cuda.empty_cache()"

# 또는 환경 변수 설정 (이미 sugar에 적용됨)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 9. Troubleshooting

### 9.1 GPU 인식 문제
```bash
# CDI 재생성
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker

# 테스트
docker compose run --rm pipeline nvidia-smi
```

### 9.2 권한 문제
```bash
# data 디렉토리 권한 설정
sudo chown -R $USER:$USER ./data
chmod -R 755 ./data
```

### 9.3 컨테이너 로그 확인
```bash
# 특정 서비스 로그
docker compose logs pipeline

# 실시간 로그
docker compose logs -f sugar

# 최근 100줄
docker compose logs --tail=100 pipeline
```

### 9.4 컨테이너 재시작
```bash
# 특정 서비스 재시작
docker compose restart pipeline

# 모든 서비스 재시작
docker compose restart
```

### 9.5 이미지 정리
```bash
# 사용하지 않는 이미지 삭제
docker image prune

# 전체 정리 (주의)
docker system prune -a
```

## 10. 유용한 명령어 모음

### 10.1 상태 확인
```bash
# 실행 중인 컨테이너
docker compose ps

# 이미지 목록
docker compose images

# 네트워크 정보
docker network ls | grep pipeline
```

### 10.2 리소스 사용량
```bash
# 컨테이너별 리소스 사용량
docker stats $(docker compose ps -q)
```

### 10.3 빠른 진단
```bash
# 전체 상태 확인
cat << 'EOF' | bash
echo "=== Docker Compose Services ==="
docker compose ps
echo ""
echo "=== GPU Access Test ==="
docker compose run --rm pipeline python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo ""
echo "=== Data Directory ==="
ls -lh ./data/
EOF
```

## 11. 참고사항

- 모든 컨테이너는 CUDA 13.0을 지원합니다
- Pipeline 컨테이너는 PyTorch 2.9 기반입니다
- SuGaR 컨테이너는 Python 3.10 (Open3D 호환성)을 사용합니다
- 네트워크는 `pipeline_net` 브리지로 격리되어 있습니다
- GPU는 모든 컨테이너에서 공유됩니다 (동시 실행 시 메모리 주의)

---

**SNUKDT 11기 코오롱 모빌리티그룹 캡스톤 프로젝트 팀**