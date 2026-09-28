# SNUKDT 11기 코오롱 모빌리티그룹 캡스톤 프로젝트: CUDA Usability Guide
해당 하드웨어 DGX spark 환경에서 CUDA를 활용하여 GPU 가속을 최적화하는 방법에 대한 가이드입니다.

## 기본 환경 구성
GB10에서는 CUDA 13.0을 사용하고, 현재 Driver version 580.95.05, CUDA Toolkit version 13.0.88이 설치되어 있습니다.

## 1. CUDA 설치 확인
먼저, CUDA가 시스템에 올바르게 설치되어 있는지 확인합니다. 터미널에서 다음 명령어를 실행하세요:
```bash
nvcc --version
```
이 명령어는 설치된 CUDA 버전을 출력합니다. 설치되어 있지 않다면, NVIDIA 공식 웹사이트에서 CUDA 툴킷을 다운로드하여 설치하세요.

## 2. NVIDIA 드라이버 확인
CUDA가 제대로 작동하려면 최신 NVIDIA 드라이버가 필요합니다. 다음 명령어로 드라이버 버전을 확인하세요:
```bash
nvidia-smi
```
출력된 정보에서 드라이버 버전과 GPU 상태를 확인할 수 있습니다.

## 3. 환경 변수 설정
CUDA를 사용할 때 환경 변수를 올바르게 설정하는 것이 중요합니다. 다음 명령어를 사용하여 환경 변수를 설정하세요:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
이 설정은 CUDA 바이너리와 라이브러리를 시스템 경로에 추가합니다.

**영구 설정 방법** (재부팅 후에도 유지):
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

현재 환경 변수 확인:
```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

## 4. CUDA 샘플 코드 실행
CUDA 설치가 제대로 되었는지 확인하기 위해 샘플 코드를 실행해보세요. 

**주의**: 최신 CUDA 버전에서는 샘플 코드가 별도 저장소로 이동되었습니다. 다음과 같이 확인할 수 있습니다:

```bash
# 샘플 코드가 설치되어 있는 경우
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

샘플이 없는 경우, 다음 명령어로 GPU 정보를 확인할 수 있습니다:
```bash
# nvidia-smi로 GPU 정보 확인
nvidia-smi

# 또는 Python으로 확인
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 5. Docker에서 CUDA 사용
Docker 컨테이너에서 CUDA를 사용하려면 NVIDIA Container Toolkit을 설치해야 합니다. 

### 5.1 NVIDIA Container Toolkit 설치
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 5.2 Docker 런타임 설정
Docker daemon을 NVIDIA 런타임으로 구성합니다:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

설정 확인:
```bash
cat /etc/docker/daemon.json
```

### 5.3 CDI (Container Device Interface) 구성 생성
리부팅 후 GPU 인식 문제가 발생할 경우:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

### 5.4 GPU 접근 테스트
두 가지 방법으로 컨테이너에서 GPU를 사용할 수 있습니다:

**방법 1**: `--gpus all` 옵션 사용 (일반적인 방법)
```bash
docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi
```

**방법 2**: `--device` 옵션 사용 (CDI 방식, 리부팅 후 권장)
```bash
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi
```

실제 프로젝트 이미지로 테스트:
```bash
docker run --rm --device nvidia.com/gpu=all 3d-reconstruction-pipeline:latest \
  python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## 6. Troubleshooting

### 6.1 CUDA 관련 오류
**증상**: CUDA 라이브러리를 찾을 수 없다는 오류
```bash
# 해결 방법: 환경 변수 확인 및 재설정
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 6.2 드라이버 호환성 문제
**증상**: CUDA 버전과 드라이버 버전 불일치
```bash
# 드라이버 버전 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# CUDA 호환성 확인
cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA version file not found"
```

**참고**: Driver 580.95.05는 CUDA 13.0을 지원합니다.

### 6.3 Docker에서 GPU 인식 문제
**증상**: `WARNING: The NVIDIA Driver was not detected`

**해결 방법**:
```bash
# 1. CDI 구성 재생성
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 2. Docker 재시작
sudo systemctl restart docker

# 3. --device 옵션으로 테스트
docker run --rm --device nvidia.com/gpu=all 3d-reconstruction-pipeline:latest nvidia-smi

# 4. 여전히 안되면 Docker daemon 설정 확인
cat /etc/docker/daemon.json

# 5. 필요시 재구성
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 6.4 권한 문제
**증상**: `/dev/nvidia*` 접근 권한 오류
```bash
# 디바이스 권한 확인
ls -la /dev/nvidia*

# docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker
```

### 6.5 메모리 부족 오류
**증상**: `CUDA out of memory`
```bash
# GPU 메모리 사용량 확인
nvidia-smi

# Python에서 메모리 정리
python3 -c "import torch; torch.cuda.empty_cache()"

# Docker에서 shared memory 증가
docker run --shm-size=64gb --ipc=host ...
```

### 6.6 빠른 진단 스크립트
```bash
# 전체 CUDA 환경 확인
cat << 'EOF' | bash
echo "=== NVIDIA Driver ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""
echo "=== CUDA Version ==="
nvcc --version | grep release
echo ""
echo "=== Docker GPU Test ==="
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader || echo "Failed"
echo ""
echo "=== CDI Configuration ==="
ls -lh /etc/cdi/nvidia.yaml 2>/dev/null || echo "CDI not configured"
EOF
```

이 가이드를 따르면 DGX spark 환경에서 CUDA를 효과적으로 활용할 수 있습니다. 추가적인 도움이 필요하면 NVIDIA 공식 문서나 커뮤니티 포럼을 참조하세요.
**SNUKDT 11기 코오롱 모빌리티그룹 캡스톤 프로젝트 팀**