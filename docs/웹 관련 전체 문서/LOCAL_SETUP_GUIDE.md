# SNUKDT 11기 코오롱 모빌리티그룹 캡스톤 프로젝트: 로컬 실행 가이드 (Local Setup Guide)

이 문서는 로컬 환경에서 Frontend, Backend, Qdrant를 직접 실행하는 방법을 단계별로 안내합니다.

## 📋 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [환경 변수 설정](#2-환경-변수-설정)
3. [Qdrant 실행 및 데이터 로드](#3-qdrant-실행-및-데이터-로드)
4. [Redis 실행 (선택사항)](#4-redis-실행-선택사항)
5. [Backend 실행](#5-backend-실행)
6. [Frontend 실행](#6-frontend-실행)
7. [검증 및 테스트](#7-검증-및-테스트)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 사전 요구사항

### 1.1 필수 소프트웨어

- **Python**: 3.10 이상
- **Node.js**: 20 이상
- **Docker**: Qdrant 실행용 (또는 Qdrant를 직접 설치)

### 1.2 설치 확인

```bash
# Python 버전 확인
python --version  # 또는 python3 --version

# Node.js 버전 확인
node --version

# Docker 버전 확인 (Qdrant 실행용)
docker --version
```

### 1.3 프로젝트 구조 확인

```bash
cd ~/Desktop/work/3d-sales-agent/app/web-client

# 프로젝트 구조 확인
ls -la
# web-client/ 디렉토리가 있어야 함
# chatbot/ 디렉토리가 있어야 함 (데이터 소스)
```

---

## 2. 환경 변수 설정

### 2.1 Backend 환경 변수 설정

```bash
cd web-client/backend

# .env 파일 생성
# 본인 API KEY 값으로 변경하시면 됩니다.
cat > .env << 'EOF'
# OpenAI API Key (필수)
OPENAI_API_KEY=sk-your-api-key-here

# LLM Provider
LLM_PROVIDER=openai

# Database (로컬 개발 시 선택사항)
POSTGRES_USER=kolon_user
POSTGRES_PASSWORD=changeme
POSTGRES_DB=kolon_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis (Session Management) - 선택사항
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Qdrant (로컬 실행)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=kolon_used_cars

# Application
USE_AUTOENCODER=true
LOG_LEVEL=INFO
TEST_MODE=true

# 3D 모델 경로 (프로젝트 루트 기준)
CF3_PLY_PATH=../source/cf3_demo/cf3.ply
AUTOENCODER_PATH=../source/cf3_demo/autoencoder.pth
EOF
```

**중요**: `OPENAI_API_KEY`는 반드시 실제 API 키로 변경해야 합니다.

### 2.2 Frontend 환경 변수 설정

```bash
cd ~/Desktop/work/3d-sales-agent/app/web-client/frontend

# .env.local 파일 생성
cat > .env.local << 'EOF'
NEXT_PUBLIC_DEV_MODE=true

# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Gaussian ply file path
NEXT_PUBLIC_GS_PLY_PATH=/static/web-client/source/toycar/point_cloud.ply

NEXT_PUBLIC_STATIC_BASE_URL=http://localhost:8000
EOF
```

---

## 3. Qdrant 실행 및 데이터 로드

### 3.1 Qdrant 실행 (Docker 사용)

```bash
# backend 경로에서 실행
cd ~/Desktop/work/3d-sales-agent/app/web-client/backend

# 가상 환경 실행
source kolon-venv/bin/activate

# 실행
docker run -p 6333:6333 qdrant/qdrant
```

### 3.2 Qdrant 데이터 로드
~/Desktop/work/3d-sales-agent/app/web-client/source 폴더 안의 문서를 확인하세요.

### 3.3 Qdrant 데이터 확인

```bash
# 컬렉션 목록 확인
curl http://localhost:6333/collections

# 특정 컬렉션 정보 확인
curl http://localhost:6333/collections/kolon_used_cars

# 포인트 개수 확인
curl http://localhost:6333/collections/kolon_used_cars | grep -o '"points_count":[0-9]*'
```

또는 Python으로 확인:

```bash
cd web-client/backend
python -c "
from rag.rag import get_rag_pipeline
rag = get_rag_pipeline()
count = rag.client.count(collection_name='kolon_used_cars', exact=True)
print(f'📊 총 포인트 개수: {count.count}')
"
```

---

## 5. Backend 실행

### 5.1 의존성 설치

```bash
cd ~/Desktop/work/3d-sales-agent/app/web-client/backend

# 가상환경 실행 필수!
source kolon-venv/bin/activate

```

### 5.2 Backend 서버 실행

```bash
# backend 디렉토리에서 실행
cd ~/Desktop/work/3d-sales-agent/app/web-client/backend

# 개발 모드로 실행 (자동 리로드)
uvicorn main:app --reload --port 8000

```

### 5.3 Backend 확인

```bash
# 헬스체크
curl http://localhost:8000/api/health
```

예상 출력:
```json
{"status": "healthy"}
```

---

## 6. Frontend 실행

### 6.1 의존성 설치

```bash
cd web-client/frontend

# 의존성 설치
npm install
```

### 6.2 Frontend 개발 서버 실행

```bash
# frontend 디렉토리에서 실행
cd web-client/frontend

# 개발 서버 실행
npm run dev

```

### 6.3 Frontend 확인

브라우저에서 `http://localhost:3000` 접속하여 확인합니다.

---

## 8. 트러블슈팅

### 8.1 Qdrant 연결 실패

**증상:**
```
ConnectionError: Failed to connect to Qdrant
```

**해결:**
```bash
# 1. Qdrant 컨테이너 상태 확인
docker ps | grep qdrant

# 2. Qdrant 재시작
docker-compose restart qdrant

# 3. 포트 확인
netstat -an | grep 6333  # macOS/Linux
# 또는
lsof -i :6333  # macOS

# 4. 환경 변수 확인
echo $QDRANT_HOST  # localhost여야 함
echo $QDRANT_PORT  # 6333이어야 함
```

### 8.2 데이터 로드 실패

**증상:**
```
FileNotFoundError: No *_vector.json files found
```

**해결:**
```bash
# 1. 데이터 디렉토리 확인
ls -la chatbot/car_data/*_vector.json

# 2. 데이터 디렉토리 경로 확인
# data_loader.py는 프로젝트 루트 기준으로 경로를 계산합니다
# 절대 경로로 지정해보세요:
python -m rag.data_loader \
  --data-dir /Users/tykim/Desktop/work/SNU_KDT_KOLON_CAPSTONE_PROJECT/chatbot/car_data \
  --clear-existing
```

### 8.3 Backend 실행 실패

**증상:**
```
ModuleNotFoundError: No module named 'xxx'
```

**해결:**
```bash
# 1. 가상환경 활성화 확인
which python  # venv/bin/python을 가리켜야 함

# 2. 의존성 재설치
pip install -r requirements.txt

# 3. Python 경로 확인
python --version  # 3.10 이상이어야 함
```

### 8.4 Frontend 실행 실패

**증상:**
```
Error: Cannot find module 'xxx'
```

**해결:**
```bash
# 1. node_modules 삭제 후 재설치
rm -rf node_modules
npm install

# 2. npm 버전 확인
npm --version  # 4.9.2 이상 권장

# 3. Node.js 버전 확인
node --version  # 20 이상 권장
```

### 8.5 API 호출 실패 (CORS 에러)

**증상:**
```
Access to fetch at 'http://localhost:8000/api/...' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**해결:**
```bash
# Backend의 main.py에서 CORS 설정 확인
# allow_origins에 "http://localhost:3000"이 포함되어 있어야 함

# 또는 .env.local에서 API URL 확인:
# NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 8.7 포트 충돌

**증상:**
```
Address already in use
```

**해결:**
```bash
# 1. 포트 사용 중인 프로세스 확인
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
lsof -i :6333  # Qdrant

# 2. 프로세스 종료
kill -9 <PID>

# 3. 또는 다른 포트 사용
# Backend: uvicorn main:app --port 8001
# Frontend: npm run dev --port 3001
```

---

## 9. 실행 순서 요약

로컬 환경에서 전체 시스템을 실행하는 권장 순서:

```bash
# 1. Qdrant 실행
cd ~/Desktop/work/3d-sales-agent/app/web-client/backend
uvicorn main:app --reload --port 8000

# 2. Qdrant 데이터 로드
# 위에서 설명한대로 source/ 경로의 문서 내용을 확인할 것

# 3. Backend 실행 (새 터미널)
cd ~/Desktop/work/3d-sales-agent/app/web-client/backend
source kolon-venv/bin/activate
python -m uvicorn main:app --reload --port 8000

# 4. Frontend 실행 (새 터미널)
cd ~/Desktop/work/3d-sales-agent/app/web-client/frontend
npm run dev
```

---

## 10. 유용한 명령어 모음

### Qdrant 관리

```bash
# Qdrant 컬렉션 목록
curl http://localhost:6333/collections

# 컬렉션 삭제
curl -X DELETE http://localhost:6333/collections/kolon_used_cars

# 컬렉션 정보
curl http://localhost:6333/collections/kolon_used_cars
```

---

## 11. 다음 단계

로컬 환경이 정상적으로 실행되면:

1. **개발 시작**: `HANDOVER_DOCUMENT.md`의 개발 가이드 참조
2. **프로덕션 배포**: `HANDOVER_DOCUMENT.md`의 배포 섹션 참조
3. **문서 확인**: 프로젝트 내 다른 문서들 확인

---

**문서 버전**: 1.0.0  
**최종 업데이트**: 2026-01-20
**관련 문서**: `HANDOVER_DOCUMENT.md`
