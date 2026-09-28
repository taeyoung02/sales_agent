## SNU KDT KOLON Capstone - Web Client

`web-client`는 3D 차량 시뮬레이터와 AI 딜러 챗봇을 제공하는 **프론트엔드 + 백엔드 통합 프로젝트**입니다.  
이 문서는 **로컬 환경에서 직접 실행**하는 방법을 중심으로 설명합니다.

---

## 구성 요소

- **Frontend**
  - 위치: `web-client/frontend`
  - 기술 스택: Next.js (React), TypeScript
  - 역할: 3D 차량 뷰어, 프레젠테이션 플레이어, 채팅 UI

- **Backend**
  - 위치: `web-client/backend`
  - 기술 스택: FastAPI (Python)
  - 역할: 챗봇, RAG 검색, 3D 관련 API, 세션 관리

- **외부/로컬 서비스**
  - **Qdrant**: 벡터 데이터베이스 (중고차 정보 임베딩 저장)
  - **Redis**: 세션 관리 (선택 사항, 있으면 더 안정적인 세션 관리)

로컬에서 전체 시스템을 띄우려면 **Frontend + Backend + Qdrant**가 필요합니다.

---

## 개발 환경 요구사항

- **Node.js**: 20 이상
- **npm** 또는 **yarn**
- **Python**: 3.10 이상 (백엔드용)
- **Docker**: Qdrant 실행용(권장)

버전 확인:

```bash
node --version
npm --version
python --version      # 또는 python3 --version
docker --version
```

---

## 로컬 환경 변수 설정

### 1. Backend (`web-client/backend/.env`)

```bash
cd web-client/backend

cat > .env << 'EOF'
# OpenAI API Key (필수)
OPENAI_API_KEY=sk-your-api-key-here

# LLM Provider
LLM_PROVIDER=openai

# Database (로컬 개발 시 선택 사항)
POSTGRES_USER=kolon_user
POSTGRES_PASSWORD=changeme
POSTGRES_DB=kolon_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis (세션 관리, 선택 사항)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Qdrant (로컬 Qdrant 컨테이너 기준)
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

> **주의**: `LLM API KEY`는 반드시 실제 키로 교체해야 합니다.

### 2. Frontend (`web-client/frontend/.env.local`)

```bash
cd web-client/frontend

cat > .env.local << 'EOF'
NEXT_PUBLIC_DEV_MODE=true

# Backend API URL (로컬 백엔드)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Gaussian ply file path (정적 리소스 경로)
NEXT_PUBLIC_GS_PLY_PATH=/static/web-client/source/toycar/point_cloud.ply

# 정적 파일을 제공하는 백엔드 베이스 URL
NEXT_PUBLIC_STATIC_BASE_URL=http://localhost:8000
EOF
```

---

## Qdrant 실행 (로컬)

Qdrant는 벡터 검색을 위한 필수 서비스입니다. Docker로 실행하는 것을 권장합니다.
백엔드 서버를 먼저 실행한 상태에서 진행해주세요.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

실행 확인:

```bash
curl http://localhost:6333/collections
```

### Qdrant에 중고차 벡터 데이터 적재하기

1. **Qdrant 컨테이너 실행**

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **백엔드 가상환경(선택) 활성화 후, 데이터 로더 실행**

   ```bash
   cd /Users/tykim/Desktop/work/SNU_KDT_KOLON_CAPSTONE_PROJECT/web-client/backend

   # (선택) 가상환경 활성화
   # source venv/bin/activate

   # chatbot/car_data 안의 *_vector.json 파일들을 Qdrant에 적재
   python -m rag.data_loader \
     --data-dir /Users/tykim/Desktop/work/SNU_KDT_KOLON_CAPSTONE_PROJECT/web-client/source/new_car_data \
     --clear-existing
   ```

3. **적재 결과 확인**

   ```bash
   # 컬렉션 목록 확인
   curl http://localhost:6333/collections

   # 또는 Python 스크립트로 포인트 개수 확인
   cd /Users/tykim/Desktop/work/SNU_KDT_KOLON_CAPSTONE_PROJECT/web-client/backend
   python check_qdrant_count.py
   ```

정상적으로 적재되었다면 `kolon_used_cars` 컬렉션이 생성되고, Backend의 RAG 검색/챗봇 기능에서 이 데이터를 사용하게 됩니다.

---

## Backend 실행 방법

### 1. 의존성 설치

```bash
cd web-client/backend

# (선택) 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 서버 실행

```bash
cd web-client/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

헬스 체크:

```bash
curl http://localhost:8000/api/health
```

---

## Frontend 실행 방법

### 1. 의존성 설치

```bash
cd web-client/frontend
npm install   # 또는 yarn
```

### 2. 개발 서버 실행

```bash
cd web-client/frontend
npm run dev   # 또는 yarn dev
```

브라우저에서 `http://localhost:3000`으로 접속하면 3D 뷰어 및 챗봇 UI를 확인할 수 있습니다.

---

## 로컬 실행 순서 요약

1. **Backend 실행**
   - `cd web-client/backend`
   - (필요 시 가상환경 활성화 후) `uvicorn main:app --reload --port 8000`
2. **Qdrant 실행**
   - `docker run -p 6333:6333 qdrant/qdrant`
   - 데이터 로드: 
   ```
   python -m rag.data_loader \
      --data-dir /Users/tykim/Desktop/work/SNU_KDT_KOLON_CAPSTONE_PROJECT/web-client/source/new_car_data \
      --clear-existing
   ```
3. **Frontend 실행**
   - `cd web-client/frontend`
   - `npm run dev`
4. 브라우저에서 `http://localhost:3000` 접속

자세한 트러블슈팅 및 고급 설정은 `web-client/LOCAL_SETUP_GUIDE.md`를 참고하세요.

---


## 전체 프롬프트 
1. 경로 `web-client/backend/prompts/`
   - `base.py`: 공통 프롬프트 정의
   - `tools.py`: Tool definitions for LLM function calling
   - `evaluator_role.py`: Evaluator Agent 역할 프롬프트
   - `planner_lead_classification.py`: Planner Lead Classification 프롬프트
   - `planner_role.py`: Planner Agent 역할 프롬프트
   - `sales_knowledge.py`: 세일즈 지식 프롬프트
   - `tool_guidelines.py`: 전체 Tool 가이드라인 프롬프트



## 기타 문서

- `web-client/LOCAL_SETUP_GUIDE.md` : 로컬 환경 전체 세팅(Frontend/Backend/Qdrant/Redis) 상세 가이드
- `web-client/HANDOVER_DOCUMENT.md` : 아키텍처, 기능 설명, 배포 가이드 등 종합 문서
- `web-client/backend/DEPLOYMENT_GUIDE.md` : 백엔드 배포 가이드 (클라우드/서버 기준)



## 프로젝트 구조
```
├── backend                   # 백엔드
│   ├── Dockerfile
│   ├── README.md
│   ├── main.py               # Root file
│   ├── middleware      
│   ├── prompts               # 프롬프트 관련
│   ├── rag                   # RAG 
│   ├── requirements.txt      # used packages
│   ├── routers               # FastAPI routers
│   ├── schemas               # Type schemas
│   ├── services              # LLM agent - planner, evaluator, executor agent
│   └── utils                 # etc utils
│
├── docker-compose.prod.yml   # 운영용 docker compose
├── docker-compose.yml        # 로컬용 docker compose
├── frontend                  # 프론트엔드
│   ├── Dockerfile
│   ├── README.md
│   ├── package.json
│   └── src                   
├── nginx                     # nginx 관련
│   ├── certs
│   ├── conf.d
│   ├── logs
│   └── nginx.conf
├── scripts                   # 스크립트 파일
│   ├── README.md
│   ├── check_qdrant.sh
│   ├── init_qdrant.sh
│   └── update_qdrant.sh
└── source
   ├── new_car_data          # vector DB 적재용 차량 데이터
   ├── super_car             # 테스트용 3D 차량 데이터
   ├── test                  # 테스트용 3D 차량 데이터
   └── toycar                # 테스트용 3D 차량 데이터

```