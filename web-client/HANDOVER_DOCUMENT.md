# Web-Client 프로젝트 인수인계 문서

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [개발 환경 설정](#3-개발-환경-설정)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [주요 기능 및 비즈니스 로직](#5-주요-기능-및-비즈니스-로직)
6. [데이터베이스 및 스토리지](#6-데이터베이스-및-스토리지)
7. [배포 및 운영](#7-배포-및-운영)
8. [개발 가이드](#8-개발-가이드)
9. [트러블슈팅](#9-트러블슈팅)
10. [참고 자료](#10-참고-자료)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

**3D-RAG Chatbot 웹 통합 서비스**는 자동차 판매 상담을 위한 AI 챗봇과 3D 차량 시뮬레이터를 통합한 웹 애플리케이션입니다.

**주요 목표:**
- 사용자 질의에 대한 RAG(Retrieval-Augmented Generation) 기반 답변 제공
- 3D 차량 모델을 실시간으로 조작하고 시각화
- 세션 기반 대화 관리 및 자동 로그 저장
- 히트맵 기반 차량 부위 검색 기능

### 1.2 기술 스택

#### Backend
- **Framework**: FastAPI 0.121.2
- **Python**: 3.10+
- **LLM**: gpt-4.1-mini (채팅), gpt-4o-mini(간단 작업용), text-embedding-3-small (임베딩)
- **Vector DB**: Qdrant (로컬 또는 Cloud)
- **Session Storage**: Redis 7

#### Frontend
- **Framework**: Next.js 16.1.1
- **React**: 19.2.0
- **3D 렌더링**: Three.js, @react-three/fiber, @react-three/drei
- **3D 포맷**: Spark Splat (Gaussian Splatting)
- **State Management**: Zustand
- **UI**: Tailwind CSS, Radix UI

#### Infrastructure
- **Container**: Docker, Docker Compose
- **Reverse Proxy**: Nginx
- **Tunnel**: Cloudflare Tunnel (프로덕션)

### 1.3 주요 기능

1. **AI 챗봇**
   - 3단계 Agent 파이프라인 (Planner → Evaluator → Executor)
   - RAG 기반 차량 정보 검색
   - 도구 호출 (카메라 제어, 색상 변경, 부위 삭제 등)
   - Evidence 패널로 출처 표시

2. **3D 뷰어**
   - Gaussian Splatting 기반 실시간 렌더링
   - 카메라 프리셋 및 애니메이션
   - 히트맵 기반 부위 검색
   - 색상 변경, 부위 삭제 등 편집 기능

3. **세션 관리**
   - Redis 기반 세션 TTL 관리 (30분)
   - 자동 만료 시 로그 저장 (JSON)
   - 쿠키 기반 세션 ID (HttpOnly, SameSite=Lax)

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처

```
┌─────────────┐
│   Browser   │
│  (Next.js)  │
└──────┬──────┘
       │ HTTP/HTTPS
       ↓
┌─────────────┐
│    Nginx    │  Reverse Proxy
│  (Port 80)  │  - Static files
└──────┬──────┘  - API routing
       │
       ├──────────────┬──────────────┐
       ↓              ↓              ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Frontend   │ │   Backend   │ │   Static    │
│  (Next.js)  │ │  (FastAPI)  │ │   Files     │
│  Port 3000  │ │  Port 8000  │ │  (3D PLY)   │
└─────────────┘ └──────┬──────┘ └─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  PostgreSQL │ │    Qdrant   │ │    Redis    │
│  Port 5432  │ │  Port 6333  │ │  Port 6379  │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 2.2 채팅 플로우

```
1. 사용자 입력 (프론트엔드)
   ↓
2. 히트맵 요청 감지 → 히트맵 API 호출 (별도 처리)
   또는
   일반 메시지 → /api/chat 또는 /api/stream POST 요청
   ↓
3. 백엔드 라우터 (chat.py)
   ↓
4. LLM 클라이언트 (llm_client.py)
   ├─ RAG 파이프라인 검색 (rag.py)
   │  ├─ 쿼리 임베딩 생성
   │  ├─ Qdrant 벡터 DB 검색
   │  └─ 관련 문서 반환
   ├─ 컨텍스트 구성 (시스템 프롬프트 + RAG 결과)
   ├─ Planner - Evaluator - Executor 순 진행
   ├─ 도구 호출 처리 (필요시)
   └─ 응답 생성
   ↓
5. 백엔드 라우터에서 도구 실행 (카메라 제어 등)
   ↓
6. ChatResponse 반환 (response, tool_calls, evidence)
   ↓
7. 프론트엔드 결과 처리
   ├─ 도구 호출 실행 (3D 뷰어 제어 - 3D 차량, TTS 영상)
   └─ 메시지 표시
```

### 2.3 세션 관리 아키텍처

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │ HTTP (Cookie: session_id)
       ↓
┌─────────────┐
│    Nginx    │  keepalive: 65s << 1800s (세션 TTL)
│ (Reverse    │  proxy_set_header Host $host (필수!)
│  Proxy)     │  API caching: OFF
└──────┬──────┘
       ↓
┌─────────────────────┐
│  FastAPI Backend    │
│  ┌───────────────┐  │
│  │ SessionMiddle │  │  1. 세션 ID 추출/생성
│  │     ware      │  │  2. Redis TTL 갱신
│  └───────┬───────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │  Chat Router  │  │  3. 대화 로그 생성
│  └───────┬───────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │SessionLogStore│  │  4. 메모리에 임시 저장
│  └───────────────┘  │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│       Redis         │
│  session:{id}       │  TTL: 1800s (30분)
│  notify-keyspace-   │  Config: Ex 활성화
│  events: Ex         │
└─────────┬───────────┘
          │ TTL 만료 이벤트
          ↓ PUBLISH __keyevent@0__:expired
┌─────────────────────┐
│SessionExpiryListener│  5. 만료 이벤트 구독
│  (Background Task)  │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  SessionLogStore    │  6. 로그 파일 저장
│   .save_to_file()   │
└─────────┬───────────┘
          ↓
┌─────────────────────────────────────────┐
│  backend/logs/session_logs/             │
│    session_log_{session_id}.json        │  7. JSON 파일 생성
└─────────────────────────────────────────┘
```

---

## 3. 개발 환경 설정

### 3.1 사전 요구사항

- **OS**: macOS, Linux (Ubuntu 20.04+ 권장)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.10+ (로컬 개발 시)
- **Node.js**: 20+ (로컬 개발 시)
- **Yarn**: 4.9.2+ (프론트엔드 패키지 매니저)

### 3.2 로컬 개발 환경 설정

#### 3.2.1 Redis 설정 (로컬 vs Docker)

**로컬 개발 환경 (Docker 없이)**

```bash
# 1. Redis 설치 (macOS)
brew install redis

# 2. Redis 시작 (keyspace notification 활성화)
redis-server --notify-keyspace-events Ex

# 3. 환경 변수 설정
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=  # 로컬에서는 비밀번호 없음

# 또는 .env 파일에 추가 (backend/.env)
cat >> backend/.env << 'EOF'

# Redis (로컬 개발)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
EOF
```

**Docker 환경**

```bash
cd web-client

# 개발 환경
docker-compose up -d

# 프로덕션 환경
docker-compose -f docker-compose.prod.yml up -d
```

#### 3.2.2 Backend 설정

```bash
cd web-client/backend

# 1. 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env  # .env.example이 있다면
# 또는 직접 생성
cat > .env << 'EOF'
# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# LLM Provider
LLM_PROVIDER=openai

# Database
POSTGRES_USER=kolon_user
POSTGRES_PASSWORD=changeme
POSTGRES_DB=kolon_db
POSTGRES_HOST=localhost  # 로컬 개발
POSTGRES_PORT=5432

# Redis (Session Management)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=kolon_used_cars

# Application
USE_AUTOENCODER=true
LOG_LEVEL=INFO
TEST_MODE=true
EOF

# 4. Backend 실행 - /backend 경로에서 실행
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3.2.3 Frontend 설정

```bash
cd web-client/frontend

# 1. 의존성 설치
yarn install

# 2. 환경 변수 설정
# .env.local 파일 생성 (선택사항)
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000/api
EOF

# 3. 개발 서버 실행
yarn dev
```

#### 3.2.4 Docker Compose로 전체 실행

```bash
cd web-client

# 개발 환경
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 상태 확인
docker-compose ps
```

### 3.3 환경 변수 목록

#### Backend (.env)

| 변수명 | 설명 | 기본값 | 필수 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API 키 | - | ✅ |
| `LLM_PROVIDER` | LLM 제공자 (openai) | openai | ✅ |
| `POSTGRES_HOST` | PostgreSQL 호스트 | localhost | ✅ |
| `POSTGRES_PORT` | PostgreSQL 포트 | 5432 | ✅ |
| `POSTGRES_USER` | PostgreSQL 사용자 | kolon_user | ✅ |
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 | - | ✅ |
| `POSTGRES_DB` | PostgreSQL 데이터베이스 | kolon_db | ✅ |
| `REDIS_HOST` | Redis 호스트 | localhost | ✅ |
| `REDIS_PORT` | Redis 포트 | 6379 | ✅ |
| `REDIS_DB` | Redis DB 번호 | 0 | ✅ |
| `REDIS_PASSWORD` | Redis 비밀번호 | - | ⚠️ (프로덕션 필수) |
| `QDRANT_HOST` | Qdrant 호스트 | localhost | ✅ |
| `QDRANT_PORT` | Qdrant 포트 | 6333 | ✅ |
| `QDRANT_COLLECTION` | Qdrant 컬렉션명 | kolon_used_cars | ✅ |
| `QDRANT_URL` | Qdrant Cloud URL (선택) | - | - |
| `QDRANT_API_KEY` | Qdrant Cloud API 키 (선택) | - | - |
| `USE_AUTOENCODER` | Autoencoder 사용 여부 | true | - |
| `LOG_LEVEL` | 로그 레벨 | INFO | - |
| `TEST_MODE` | 테스트 모드 | true | - |
| `CF3_PLY_PATH` | CF3 PLY 파일 경로 | /backend/source/cf3_demo/cf3.ply | - |
| `AUTOENCODER_PATH` | Autoencoder 모델 경로 | /backend/source/cf3_demo/autoencoder.pth | - |
| `TUNNEL_TOKEN` | Cloudflare Tunnel 토큰 (프로덕션) | - | - |

#### Frontend (.env.local)

| 변수명 | 설명 | 기본값 | 필수 |
|--------|------|--------|------|
| `NEXT_PUBLIC_API_URL` | 백엔드 API URL | http://localhost:8000/api | ✅ |

---

## 4. 프로젝트 구조

### 4.1 디렉토리 구조

```
web-client/
├── backend/                    # FastAPI 백엔드
│   ├── main.py                 # 애플리케이션 진입점
│   ├── requirements.txt        # Python 의존성
│   ├── Dockerfile              # Docker 이미지 빌드
│   ├── .env                    # 환경 변수 (gitignore)
│   │
│   ├── routers/                # API 라우터
│   │   ├── chat.py             # 채팅 API (3단계 Agent 파이프라인)
│   │   ├── scene.py            # 3D 씬 관리
│   │   ├── convert.py          # 파일 변환
│   │   ├── heatmap.py          # 히트맵 생성
│   │   ├── data.py             # 데이터 조회
│   │   ├── prompts.py          # 프롬프트 관리
│   │   ├── camera_presets.py   # 카메라 프리셋
│   │   └── vanilla_chat.py     # 기본 채팅 (Agent 없음)
│   │
│   ├── services/               # 비즈니스 로직
│   │   ├── base_agent.py       # 기본 Agent 클래스
│   │   ├── llm_planner.py      # Planner Agent
│   │   ├── llm_evaluator.py    # Evaluator Agent
│   │   ├── llm_executor.py     # Executor Agent
│   │   └── llm/                # LLM 관련 유틸리티
│   │       ├── llm_client.py   # OpenAI 클라이언트
│   │       └── llm_usage_tracker.py  # 토큰 사용량 추적
│   │
│   ├── rag/                    # RAG 파이프라인
│   │   ├── rag.py              # RAG 검색 로직
│   │   ├── data_loader.py      # 데이터 로더
│   │   └── vin_mapper.py       # VIN 매핑
│   │
│   ├── prompts/                # 프롬프트 템플릿
│   │   ├── base.py             # 기본 프롬프트
│   │   ├── tools.py             # 도구 정의
│   │   ├── full/                # 전체 프롬프트
│   │   │   ├── planner_lead_classification.py
│   │   │   └── ...
│   │   └── compact/             # 간소화 프롬프트
│   │
│   ├── middleware/             # 미들웨어
│   │   └── session.py           # 세션 관리 미들웨어
│   │
│   ├── utils/                  # 유틸리티
│   │   ├── redis_client.py     # Redis 클라이언트
│   │   ├── session_expiry_listener.py  # 세션 만료 리스너
│   │   ├── session_log.py      # 세션 로그 저장
│   │   ├── logging_config.py    # 로깅 설정
│   │   ├── heatmap_generator.py # 히트맵 생성
│   │   └── ...
│   │
│   ├── schemas/                # Pydantic 스키마
│   │   └── schemas.py           # 요청/응답 스키마
│   │
│   ├── docs/                   # 문서
│   │   └── OPENAI_API_GUIDE.md
│   │
│   └── logs/                   # 로그 파일
│       └── session_logs/       # 세션 로그 JSON
│
├── frontend/                    # Next.js 프론트엔드
│   ├── package.json            # Node.js 의존성
│   ├── Dockerfile              # Docker 이미지 빌드
│   ├── next.config.ts          # Next.js 설정
│   │
│   └── src/
│       ├── app/                # Next.js App Router
│       │   ├── page.tsx        # 메인 페이지
│       │   ├── layout.tsx      # 레이아웃
│       │   ├── api/            # API 라우트 (프록시)
│       │   │   └── [...path]/route.ts
│       │   ├── prompts/       # 프롬프트 관리 페이지
│       │   └── vanilla/        # 기본 채팅 페이지
│       │
│       ├── components/          # React 컴포넌트
│       │   ├── chat-panel.tsx  # 채팅 패널
│       │   ├── evidence-panel.tsx  # Evidence 패널
│       │   ├── three-viewer.tsx    # 3D 뷰어
│       │   ├── spark-splat.tsx     # Spark Splat 렌더러
│       │   ├── camera-presets.tsx  # 카메라 프리셋
│       │   └── ui/             # UI 컴포넌트 (Radix UI)
│       │
│       ├── lib/                # 라이브러리
│       │   ├── api.ts          # API 클라이언트
│       │   ├── state.ts        # Zustand 상태 관리
│       │   ├── types.ts        # TypeScript 타입
│       │   └── ...
│       │
│       └── types/              # 타입 정의
│           └── car_info.types.ts
│
├── nginx/                       # Nginx 설정
│   ├── nginx.conf              # 메인 설정
│   ├── conf.d/                 # 추가 설정
│   └── certs/                  # SSL 인증서 (프로덕션)
│
├── scripts/                    # 유틸리티 스크립트
│   ├── init_qdrant.sh         # Qdrant 초기화
│   ├── update_qdrant.sh       # Qdrant 업데이트
│   └── check_qdrant.sh        # Qdrant 상태 확인
│
├── source/                     # 정적 파일 (3D 모델, PLY)
│   └── cf3_demo/
│       ├── cf3.ply
│       └── autoencoder.pth
│
├── docker-compose.yml          # 개발 환경 Docker Compose
├── docker-compose.prod.yml     # 프로덕션 환경 Docker Compose
├── docker-compose.init.yml     # 초기화용 Docker Compose
│
└── README.md                   # 기본 README
```

### 4.2 주요 파일 설명

#### Backend

**main.py**
- FastAPI 애플리케이션 진입점
- Redis 연결 및 keyspace notification 설정
- SessionExpiryListener 백그라운드 태스크 시작
- SessionMiddleware 추가
- 라우터 등록

**routers/chat.py**
- 채팅 API 엔드포인트 (`/api/chat`)
- 3단계 Agent 파이프라인 (Planner → Evaluator → Executor)
- 세션 로그 기록
- 도구 호출 처리

**services/base_agent.py**
- 기본 Agent 클래스
- 세션 관리 및 로깅
- BaseAgent cleanup (5분마다)

**utils/redis_client.py**
- Redis 클라이언트 래퍼
- 세션 메타데이터 저장/조회
- Keyspace notification 관리

**utils/session_expiry_listener.py**
- Redis keyspace notification 구독
- 세션 만료 이벤트 처리
- 자동 로그 저장 트리거

**middleware/session.py**
- 쿠키 기반 세션 ID 관리
- Redis TTL 갱신 (매 요청마다)
- HttpOnly, SameSite=Lax 쿠키 설정

#### Frontend

**src/app/page.tsx**
- 메인 페이지 (채팅 + 3D 뷰어)

**src/lib/api.ts**
- 백엔드 API 호출 래퍼 함수
- 공통 에러 처리
- 타입 안전성 보장

**src/components/chat-panel.tsx**
- 채팅 UI 컴포넌트
- 메시지 표시 및 입력
- 도구 호출 처리

**src/components/three-viewer.tsx**
- 3D 뷰어 컨테이너
- Spark Splat 렌더러 통합

**src/components/spark-splat.tsx**
- Gaussian Splatting 렌더러
- 카메라 제어
- 색상 변경, 부위 삭제 등 편집 기능

---

## 5. 주요 기능 및 비즈니스 로직

### 5.1 채팅 시스템

#### 5.1.1 3단계 Agent 파이프라인

**1. Planner Agent (`services/llm_planner.py`)**
- 사용자 메시지 분석
- 최대 3개의 실행 시나리오 생성
- 각 시나리오에 plan_id, description, tools 할당

**2. Evaluator Agent (`services/llm_evaluator.py`)**
- Planner가 생성한 시나리오 평가
- 최종 1개 시나리오 선택
- 선택 이유 제공

**3. Executor Agent (`services/llm_executor.py`)**
- 선택된 시나리오 실행
- LLM 호출 및 도구 실행
- 응답 생성

#### 5.1.2 RAG 파이프라인

**검색 프로세스:**
1. 사용자 쿼리를 임베딩으로 변환 (`text-embedding-3-small`)
2. Qdrant에서 유사도 검색 (top-k)
3. 관련 문서 추출 및 컨텍스트 구성
4. LLM에 컨텍스트와 함께 전달

**파일 위치:**
- `backend/rag/rag.py`: RAG 검색 로직
- `backend/rag/data_loader.py`: 데이터 로더

#### 5.1.3 도구 호출 (Tools)

**지원 도구:**
- `change_camera_preset`: 카메라 프리셋 변경
- `change_color`: 차량 색상 변경
- `delete_part`: 차량 부위 삭제
- `load_vehicle`: 차량 로드

**도구 정의:**
- `backend/prompts/tools.py`: 도구 스키마 정의
- `backend/routers/chat.py`: 도구 실행 로직

### 5.2 3D 뷰어 시스템

#### 5.2.1 Gaussian Splatting 렌더링

- **라이브러리**: Spark Splat (`@sparkjsdev/spark`)
- **포맷**: PLY 파일 (Gaussian Splatting)
- **렌더러**: `src/components/spark-splat.tsx`

#### 5.2.2 카메라 제어

- **프리셋**: 미리 정의된 카메라 위치/각도
- **애니메이션**: 부드러운 카메라 이동
- **API**: `/api/camera-presets`

#### 5.2.3 히트맵 기능

- **목적**: 사용자 질의에 해당하는 차량 부위 시각화
- **프로세스**:
  1. CLIP 인코더로 텍스트 임베딩 생성
  2. Autoencoder로 차량 부위별 특징 추출
  3. 유사도 계산 및 히트맵 생성
- **API**: `/api/heatmap`

### 5.3 세션 관리

#### 5.3.1 세션 생성

1. 사용자 첫 방문 시 `SessionMiddleware`가 세션 ID 생성 (UUID)
2. Redis에 세션 메타데이터 저장 (`session:{id}`)
3. TTL 설정 (1800초 = 30분)
4. 쿠키에 세션 ID 저장 (HttpOnly, SameSite=Lax)

#### 5.3.2 세션 유지

- 매 API 요청마다 `SessionMiddleware`가 Redis TTL 갱신
- 사용자 활동이 있으면 세션 만료 시간 연장

#### 5.3.3 세션 만료 및 로그 저장

1. Redis TTL이 0이 되면 `__keyevent@0__:expired` 이벤트 발생
2. `SessionExpiryListener`가 이벤트 구독
3. `SessionLogStore.save_to_file()` 호출
4. JSON 파일로 저장 (`logs/session_logs/session_log_{id}.json`)

#### 5.3.4 세션 로그 형식

```json
{
  "session_id": "abc-123-def-456",
  "conversation": [
    {
      "turn_id": "u-0",
      "speaker": "user",
      "text": "BMW X5를 보여주세요",
      "timestamp": "2026-01-17T10:30:00Z"
    },
    {
      "turn_id": "a-0",
      "speaker": "assistant",
      "text": "BMW X5 차량을 로드했습니다...",
      "timestamp": "2026-01-17T10:30:02Z",
      "lead_status": "Warm Lead"
    }
  ],
  "evidence": {
    "a-0": [
      {
        "source_id": "WBAXXX123",
        "snippet_text": "BMW X5 xDrive40i, 2023년식..."
      }
    ]
  },
  "latency_ms": {
    "a-0": 1234.5
  },
  "token_usage": {
    "a-0": {
      "input": 500,
      "output": 200,
      "total": 700
    }
  },
  "participant_id": null,
  "user_goal_segment": null
}
```

---

## 6. 데이터베이스 및 스토리지

### 6.1 Qdrant (Vector Database)

**용도:**
- 차량 정보 벡터 검색
- RAG 파이프라인에서 사용

**연결 정보:**
- 호스트: `qdrant` (Docker) 또는 `localhost` (로컬)
- 포트: `6333` (REST API), `6334` (gRPC)
- 컬렉션: `kolon_used_cars`

**초기화:**
```bash
cd web-client/scripts
./init_qdrant.sh
```

**상태 확인:**
```bash
./check_qdrant.sh
```

**업데이트:**
```bash
./update_qdrant.sh
```

### 6.2 Redis

**용도:**
- 세션 메타데이터 저장
- 세션 TTL 관리
- Keyspace notification으로 만료 이벤트 감지

**연결 정보:**
- 호스트: `redis` (Docker) 또는 `localhost` (로컬)
- 포트: `6379`
- DB: `0`

**중요 설정:**
- `notify-keyspace-events Ex`: 만료 이벤트 활성화 (필수!)
- `maxmemory 512mb`: 메모리 제한
- `maxmemory-policy allkeys-lru`: LRU 정책
- `appendonly yes`: AOF 영속성

**백업:**
```bash
# AOF 파일 백업
cp ~/Desktop/work/3d-sales-agent/data/redis/appendonly.aof ~/backups/redis_$(date +%Y%m%d).aof

# 또는 SAVE 명령어
docker exec kolon-redis redis-cli -a YOUR_PASSWORD BGSAVE
```

### 6.3 정적 파일 스토리지

**로컬:**
- 경로: `web-client/source/`
- 마운트: Docker에서 `/backend/source` (Backend), `/var/www/static/source` (Nginx)

**Cloudflare R2 (선택사항):**
- 환경 변수로 설정 가능
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_ENDPOINT_URL`

---

## 7. 배포 및 운영

### 7.1 프로덕션 환경 구조

```
~/Desktop/work/3d-sales-agent/
├── app/                          # 애플리케이션 소스 코드
│   └── web-client/               # 프로젝트 루트
│       ├── backend/
│       ├── frontend/
│       ├── source/
│       └── scripts/
├── data/                         # 영구 데이터
│   ├── postgres/                 # PostgreSQL 데이터
│   ├── qdrant/                   # Qdrant 벡터 데이터
│   └── redis/                    # Redis 세션 데이터 (AOF)
├── logs/                         # 로그 파일
│   ├── app/                      # 애플리케이션 로그
│   │   └── session_logs/         # 세션 로그 JSON
│   └── nginx/                    # Nginx 로그
└── config/                       # 설정 파일
    └── nginx/                    # Nginx 설정 및 SSL 인증서
```

### 7.2 프로덕션 배포 절차

#### 7.2.1 사전 준비

```bash
# 1. 디렉토리 생성
mkdir -p ~/Desktop/work/3d-sales-agent/{data/{postgres,qdrant,redis},logs/{app/session_logs,nginx},config/nginx/{conf.d,certs}}

# 2. 권한 설정
chmod -R 755 ~/Desktop/work/3d-sales-agent/logs
chmod -R 755 ~/Desktop/work/3d-sales-agent/data

# 3. 환경 변수 설정
cd ~/Desktop/work/3d-sales-agent/app/web-client
cat > backend/.env << 'EOF'
# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Database
POSTGRES_USER=kolon_user
POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD_HERE
POSTGRES_DB=kolon_db

# Redis (IMPORTANT: Set a strong password!)
REDIS_PASSWORD=YOUR_REDIS_PASSWORD_HERE

# Qdrant
QDRANT_COLLECTION=kolon_used_cars

# Application
USE_AUTOENCODER=true
LOG_LEVEL=INFO
TEST_MODE=false

# Cloudflare Tunnel (if used)
TUNNEL_TOKEN=your-tunnel-token-here
EOF

chmod 600 backend/.env

# 4. Nginx 설정 복사
cp nginx/nginx.conf ~/Desktop/work/3d-sales-agent/config/nginx/nginx.conf
```

#### 7.2.2 서비스 시작

```bash
cd ~/Desktop/work/3d-sales-agent/app/web-client

# 프로덕션 환경으로 실행
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 로그 확인
docker-compose -f docker-compose.prod.yml logs -f
```

#### 7.2.3 상태 확인

```bash
# 컨테이너 상태
docker-compose -f docker-compose.prod.yml ps

# 헬스체크
docker inspect kolon-backend | grep -A 10 Health
docker inspect kolon-redis | grep -A 10 Health
docker inspect kolon-postgres | grep -A 10 Health

# Redis 설정 확인
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD CONFIG GET notify-keyspace-events
# 출력: "Ex" 포함되어야 함
```

### 7.3 모니터링

#### 7.3.1 리소스 사용량

```bash
# 컨테이너 리소스 모니터링
docker stats kolon-backend kolon-redis kolon-postgres kolon-qdrant

# Redis 메모리 사용량
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD INFO memory
```

#### 7.3.2 로그 확인

```bash
# Backend 로그
docker logs -f kolon-backend | grep -i "session"

# Nginx 접근 로그
tail -f ~/Desktop/work/3d-sales-agent/logs/nginx/access.log

# Nginx 에러 로그
tail -f ~/Desktop/work/3d-sales-agent/logs/nginx/error.log

# 세션 로그 파일
ls -lh ~/Desktop/work/3d-sales-agent/logs/app/session_logs/
```

#### 7.3.3 Redis 세션 통계

```bash
# 세션 수 확인
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD --no-auth-warning eval "return #redis.call('keys', 'session:*')" 0

# Redis 데이터베이스 통계
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD INFO stats
```

### 7.4 업데이트 및 재배포

```bash
cd ~/Desktop/work/3d-sales-agent/app/web-client

# 코드 업데이트
git pull origin main

# 컨테이너 재빌드 및 재시작
docker-compose -f docker-compose.prod.yml build backend
docker-compose -f docker-compose.prod.yml up -d backend

# 로그 확인
docker logs -f kolon-backend
```

### 7.5 백업

#### 7.5.1 Redis 백업

```bash
# AOF 파일 백업
cp ~/Desktop/work/3d-sales-agent/data/redis/appendonly.aof ~/backups/redis_$(date +%Y%m%d).aof

# 또는 SAVE 명령어
docker exec kolon-redis redis-cli -a YOUR_PASSWORD BGSAVE
```

#### 7.5.2 PostgreSQL 백업

```bash
docker exec kolon-postgres pg_dump -U kolon_user kolon_db > ~/backups/postgres_$(date +%Y%m%d).sql
```

#### 7.5.3 세션 로그 백업

```bash
tar -czf ~/backups/session_logs_$(date +%Y%m%d).tar.gz \
  ~/Desktop/work/3d-sales-agent/logs/app/session_logs/
```

---

## 8. 개발 가이드

### 8.1 새로운 API 엔드포인트 추가

#### 8.1.1 라우터 생성

```python
# backend/routers/my_router.py
from fastapi import APIRouter
from schemas.schemas import MyRequest, MyResponse

router = APIRouter(prefix="/api/my", tags=["my"])

@router.post("", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    # 로직 구현
    return MyResponse(...)
```

#### 8.1.2 라우터 등록

```python
# backend/main.py
from routers import my_router

app.include_router(my_router.router)
```

#### 8.1.3 스키마 정의

```python
# backend/schemas/schemas.py
from pydantic import BaseModel

class MyRequest(BaseModel):
    field1: str
    field2: int

class MyResponse(BaseModel):
    result: str
```

### 8.2 프론트엔드에서 API 호출

#### 8.2.1 API 클라이언트 함수 추가

```typescript
// src/lib/api.ts
export async function myApiFunction(data: MyRequest): Promise<MyResponse> {
  const response = await fetch('/api/my', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error('API call failed');
  }
  
  return response.json();
}
```

#### 8.2.2 컴포넌트에서 사용

```typescript
// src/components/my-component.tsx
import { myApiFunction } from '@/lib/api';

export function MyComponent() {
  const handleClick = async () => {
    const result = await myApiFunction({ field1: 'value', field2: 123 });
    console.log(result);
  };
  
  return <button onClick={handleClick}>Click me</button>;
}
```

### 8.3 프롬프트 수정

#### 8.3.1 프롬프트 파일 위치

- `backend/prompts/full/`: 전체 프롬프트

#### 8.3.2 프롬프트 수정 예시

```python
# backend/prompts/full/planner_lead_classification.py
PLANNER_SYSTEM_PROMPT = """
You are a helpful assistant...
# 여기에 프롬프트 내용 수정
"""
```

#### 8.3.3 테스트

```bash
# 로컬에서 테스트
cd backend
python -c "from prompts.full.planner_lead_classification import PLANNER_SYSTEM_PROMPT; print(PLANNER_SYSTEM_PROMPT)"
```

### 8.4 3D 뷰어 기능 추가

#### 8.4.1 새로운 도구 추가

1. **도구 정의** (`backend/prompts/tools.py`)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "my_new_tool",
            "description": "새로운 도구 설명",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "파라미터 설명"}
                },
                "required": ["param1"]
            }
        }
    },
    # ... 기존 도구들
]
```

2. **도구 실행 로직** (`backend/routers/chat.py`)

```python
async def execute_tool(tool_name: str, arguments: dict, session_id: str):
    if tool_name == "my_new_tool":
        # 도구 실행 로직
        result = await my_new_tool_handler(arguments)
        return result
    # ... 기존 도구들
```

3. **프론트엔드 처리** (`src/components/chat-panel.tsx`)

```typescript
const handleToolCall = async (toolCall: ToolCall) => {
  if (toolCall.function.name === 'my_new_tool') {
    // 프론트엔드 처리 로직
    await handleMyNewTool(toolCall.function.arguments);
  }
  // ... 기존 도구들
};
```

### 8.5 코딩 컨벤션

#### 8.5.1 Python

- **스타일**: PEP 8 준수
- **타입 힌팅**: 가능한 모든 함수에 타입 힌팅 추가
- **문서화**: Docstring 작성 (Google 스타일)

```python
def my_function(param1: str, param2: int) -> dict:
    """
    함수 설명
    
    Args:
        param1: 파라미터 1 설명
        param2: 파라미터 2 설명
    
    Returns:
        반환값 설명
    """
    pass
```

#### 8.5.2 TypeScript

- **스타일**: ESLint 규칙 준수
- **타입**: 모든 함수에 타입 정의
- **컴포넌트**: 함수형 컴포넌트 사용

```typescript
interface MyComponentProps {
  title: string;
  count: number;
}

export function MyComponent({ title, count }: MyComponentProps) {
  return <div>{title}: {count}</div>;
}
```

---

## 9. 트러블슈팅

### 9.1 Redis 연결 실패

**증상:**
```
❌ Redis setup failed: Error 8 connecting to redis:6379.
```

**해결:**
```bash
# 1. Redis 상태 확인
docker ps | grep redis
docker logs kolon-redis

# 2. 환경 변수 확인
echo $REDIS_HOST  # "redis" (Docker) 또는 "localhost" (로컬)

# 3. Redis 재시작
docker-compose restart redis

# 4. Redis 없이 실행 (degraded mode)
# 환경 변수 설정 없이 실행 → 자동으로 degraded mode
```

### 9.2 세션 자동 저장 안됨

**증상:**
- 세션 로그 파일이 생성되지 않음

**해결:**
```bash
# 1. Keyspace notification 확인
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD CONFIG GET notify-keyspace-events
# 출력: "Ex" 포함되어야 함

# 2. 수동 설정
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD CONFIG SET notify-keyspace-events Ex

# 3. Backend 재시작
docker-compose restart backend

# 4. 로그 확인
docker logs -f kolon-backend | grep -i "session"
```

### 9.3 Qdrant 연결 실패

**증상:**
```
ConnectionError: Failed to connect to Qdrant
```

**해결:**
```bash
# 1. Qdrant 상태 확인
docker ps | grep qdrant
docker logs kolon-qdrant

# 2. Qdrant 재시작
docker-compose restart qdrant

# 3. 컬렉션 확인
curl http://localhost:6333/collections

# 4. 초기화 스크립트 실행
cd scripts
./init_qdrant.sh
```

### 9.4 프론트엔드에서 API 호출 실패

**증상:**
- CORS 에러 또는 404 에러

**해결:**
```bash
# 1. 백엔드 CORS 설정 확인 (main.py)
# allow_origins에 프론트엔드 URL 추가

# 2. Nginx 프록시 설정 확인
# nginx/nginx.conf에서 /api 경로 프록시 확인

# 3. 환경 변수 확인
echo $NEXT_PUBLIC_API_URL  # 프론트엔드에서 사용하는 API URL

# 4. 네트워크 확인
docker network ls
docker network inspect kolon-app-network
```

### 9.5 3D 뷰어가 로드되지 않음

**증상:**
- PLY 파일이 로드되지 않거나 렌더링 실패

**해결:**
```bash
# 1. PLY 파일 경로 확인
ls -lh source/cf3_demo/cf3.ply

# 2. Nginx 정적 파일 마운트 확인
docker exec kolon-nginx ls -la /var/www/static/source/

# 3. 브라우저 콘솔 확인
# Network 탭에서 PLY 파일 요청 상태 확인

# 4. CORS 설정 확인
# Nginx에서 정적 파일 CORS 헤더 확인
```

### 9.6 세션 만료가 너무 빠름

**증상:**
- 30분보다 빨리 세션이 만료됨

**해결:**
```python
# backend/middleware/session.py
SESSION_TTL = 1800  # 30분 (초 단위)

# Redis TTL 확인
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD TTL session:YOUR_SESSION_ID
```

### 9.7 로그 파일 권한 오류

**증상:**
```
Permission denied: /backend/logs/session_logs/...
```

**해결:**
```bash
# 1. 로그 디렉토리 권한 설정
sudo chmod -R 755 ~/Desktop/work/3d-sales-agent/logs
sudo chown -R 1000:1000 ~/Desktop/work/3d-sales-agent/logs

# 2. 컨테이너 재시작
docker-compose restart backend
```

### 9.8 OpenAI API 호출 실패

**증상:**
- Rate limit 또는 인증 오류

**해결:**
```bash
# 1. API 키 확인
echo $OPENAI_API_KEY  # sk-로 시작해야 함

# 2. Rate limit 확인
# OpenAI 대시보드에서 사용량 확인

# 3. 재시도 로직 확인
# backend/services/llm/llm_client.py에서 재시도 설정 확인
```

---

## 10. 참고 자료

### 10.1 프로젝트 내 문서

- **로컬 개발 가이드**: `LOCAL_DEVELOPMENT.md`
- **프로덕션 설정 가이드**: `PRODUCTION_SETUP.md`
- **Redis 세션 구현**: `REDIS_SESSION_IMPLEMENTATION.md`
- **마이그레이션 요약**: `MIGRATION_SUMMARY.md`
- **상세 변경 사항**: `DETAILED_CHANGES.md`

### 10.2 외부 문서

- **FastAPI**: https://fastapi.tiangolo.com/
- **Next.js**: https://nextjs.org/docs
- **Redis**: https://redis.io/docs/
- **Qdrant**: https://qdrant.tech/documentation/
- **OpenAI API**: https://platform.openai.com/docs
- **Three.js**: https://threejs.org/docs/
- **Spark Splat**: https://github.com/sparkjsdev/spark

### 10.3 주요 설정 파일

- **Docker Compose (개발)**: `docker-compose.yml`
- **Docker Compose (프로덕션)**: `docker-compose.prod.yml`
- **Nginx 설정**: `nginx/nginx.conf`
- **Backend 의존성**: `backend/requirements.txt`
- **Frontend 의존성**: `frontend/package.json`

### 10.4 유용한 명령어

```bash
# 전체 서비스 시작
docker-compose up -d

# 특정 서비스만 시작
docker-compose up -d backend

# 로그 확인
docker-compose logs -f backend

# 컨테이너 재시작
docker-compose restart backend

# 컨테이너 재빌드
docker-compose build backend

# 컨테이너 내부 접속
docker exec -it kolon-backend bash

# Redis CLI 접속
docker exec -it kolon-redis redis-cli -a YOUR_PASSWORD

# PostgreSQL 접속
docker exec -it kolon-postgres psql -U kolon_user -d kolon_db
```

### 10.5 연락처 및 지원

- **프로젝트 저장소**: (Git 저장소 URL)
- **이슈 트래커**: (이슈 트래커 URL)
- **문서 저장소**: (문서 저장소 URL)

---

## 부록: 체크리스트

### 개발 환경 설정 체크리스트

- [ ] Docker 및 Docker Compose 설치 확인
- [ ] Python 3.10+ 설치 확인
- [ ] Node.js 20+ 설치 확인
- [ ] Yarn 설치 확인
- [ ] `.env` 파일 생성 및 환경 변수 설정
- [ ] Redis 실행 확인 (로컬 또는 Docker)
- [ ] PostgreSQL 실행 확인 (로컬 또는 Docker)
- [ ] Qdrant 실행 확인 (로컬 또는 Docker)
- [ ] Backend 실행 확인 (`uvicorn main:app --reload`)
- [ ] Frontend 실행 확인 (`yarn dev`)
- [ ] API 문서 접속 확인 (`http://localhost:8000/docs`)

### 프로덕션 배포 체크리스트

- [ ] 디렉토리 구조 생성 (`data/`, `logs/`, `config/`)
- [ ] `.env` 파일에 실제 비밀번호 설정
- [ ] Redis 비밀번호 설정 (`REDIS_PASSWORD`)
- [ ] PostgreSQL 비밀번호 변경
- [ ] OpenAI API 키 설정
- [ ] SSL 인증서 준비 (HTTPS)
- [ ] 디렉토리 권한 확인 (`755` for logs, data)
- [ ] Redis keyspace notification 활성화 확인
- [ ] 백업 설정 (Redis AOF, PostgreSQL, 세션 로그)
- [ ] 모니터링 도구 설정 (선택 사항)
- [ ] 방화벽 설정 (포트: 80, 443만 외부 개방)
- [ ] Cloudflare Tunnel 토큰 설정 (선택 사항)

---

**문서 버전**: 1.0.0  
**최종 업데이트**: 2026-01-17  
**작성자**: (작성자 이름)
