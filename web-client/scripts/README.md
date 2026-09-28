# Qdrant 데이터 관리 스크립트

프로덕션 환경에서 Qdrant 벡터 데이터베이스를 관리하기 위한 스크립트 모음입니다.

## 스크립트 목록

### 1. `init_qdrant.sh` - 초기 데이터 로딩
**용도**: 프로덕션 배포 시 최초 데이터 로딩 또는 전체 데이터 재로딩

```bash
# 사용법
./scripts/init_qdrant.sh

# 환경변수 설정 (선택)
export DATA_DIR=/custom/path/to/car_data
export CLEAR_EXISTING=true  # 기존 데이터 삭제 여부
```

**동작**:
- Qdrant 연결 확인
- 데이터 디렉토리 검증
- 기존 컬렉션 삭제 (CLEAR_EXISTING=true인 경우)
- 새 컬렉션 생성 및 데이터 로딩

### 2. `update_qdrant.sh` - 증분 업데이트
**용도**: 기존 데이터를 유지하며 새 데이터 추가/업데이트

```bash
# 사용법
./scripts/update_qdrant.sh

# 기존 데이터 유지하며 업데이트
CLEAR_EXISTING=false ./scripts/update_qdrant.sh
```

**동작**:
- 기존 컬렉션 유지
- 새 데이터를 upsert (기존 ID는 업데이트, 새 ID는 추가)

### 3. `check_qdrant.sh` - 상태 확인
**용도**: Qdrant 상태 및 데이터 개수 확인

```bash
# 사용법
./scripts/check_qdrant.sh
```

**출력**:
- Qdrant 헬스 상태
- 컬렉션 정보
- 데이터 포인트 개수

## 프로덕션 배포 시나리오

### 시나리오 1: 최초 배포

```bash
# 1. 서비스 시작
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 2. Qdrant 초기화 (서비스 시작 후)
./scripts/init_qdrant.sh
```

### 시나리오 2: 데이터 업데이트

```bash
# 증분 업데이트 (기존 데이터 유지)
CLEAR_EXISTING=false ./scripts/update_qdrant.sh

# 전체 재로딩 (기존 데이터 삭제)
./scripts/init_qdrant.sh
```

### 시나리오 3: Docker Compose로 자동 초기화

```bash
# 초기화 컨테이너 실행 (한 번만)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.init.yml run --rm qdrant-init
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `QDRANT_HOST` | `qdrant` | Qdrant 서버 호스트 |
| `QDRANT_PORT` | `6333` | Qdrant 서버 포트 |
| `QDRANT_COLLECTION` | `kolon_used_cars` | 컬렉션 이름 |
| `DATA_DIR` | `/backend/chatbot/car_data` | 데이터 파일 디렉토리 |
| `CLEAR_EXISTING` | `true` | 기존 데이터 삭제 여부 |

## 자동화 예시

### Cron을 사용한 정기 업데이트

```bash
# 매일 새벽 2시에 데이터 업데이트
0 2 * * * cd /path/to/project/web-client && CLEAR_EXISTING=false ./scripts/update_qdrant.sh >> /var/log/qdrant-update.log 2>&1
```

### CI/CD 파이프라인 통합

```yaml
# .github/workflows/deploy.yml
- name: Initialize Qdrant
  run: |
    ssh user@dgx-spark "cd /path/to/project/web-client && ./scripts/init_qdrant.sh"
```

## 트러블슈팅

### Qdrant 연결 실패

```bash
# Qdrant 상태 확인
docker-compose ps qdrant
docker-compose logs qdrant

# 수동 연결 테스트
curl http://qdrant:6333/health
```

### 데이터 로딩 실패

```bash
# 데이터 디렉토리 확인
ls -la /backend/chatbot/car_data/

# 파일 형식 확인
head -n 5 /backend/chatbot/car_data/car_info*_vector.json
```

### 컬렉션 확인

```bash
# 컬렉션 목록 확인
docker-compose exec backend python -c "
from rag.rag import get_rag_pipeline
rag = get_rag_pipeline()
collections = rag.client.get_collections()
print([c.name for c in collections.collections])
"
```

## 주의사항

1. **초기 배포 시**: `init_qdrant.sh`는 기존 데이터를 삭제합니다. 신중하게 사용하세요.
2. **데이터 업데이트 시**: `update_qdrant.sh`는 기존 데이터를 유지하며 업데이트합니다.
3. **백업**: 데이터 로딩 전에 Qdrant 백업을 권장합니다.
4. **모니터링**: 정기적으로 `check_qdrant.sh`로 상태를 확인하세요.
