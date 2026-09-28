#!/bin/bash
# Qdrant 초기 데이터 로딩 스크립트
# 프로덕션 배포 시 실행

set -e  # 에러 발생 시 중단

echo "🚀 Starting Qdrant data initialization..."

# 환경 변수 확인
# 호스트에서 실행 시 localhost 사용, 컨테이너 내부에서는 qdrant 사용
if [ -z "$QDRANT_HOST" ]; then
    # 호스트에서 실행 중인지 확인 (docker-compose exec로 실행되는지)
    if [ -n "$DOCKER_CONTAINER" ] || docker-compose ps kolon-qdrant > /dev/null 2>&1; then
        # 컨테이너 내부에서 실행되는 경우
        export QDRANT_HOST=qdrant
    else
        # 호스트에서 실행되는 경우
        export QDRANT_HOST=localhost
    fi
    echo "📍 Using QDRANT_HOST: $QDRANT_HOST"
fi

if [ -z "$QDRANT_PORT" ]; then
    echo "📍 Using QDRANT_PORT: 6333"
    export QDRANT_PORT=6333
fi

# Qdrant 연결 확인
echo "🔍 Checking Qdrant connection at ${QDRANT_HOST}:${QDRANT_PORT}..."
if ! timeout 10 bash -c "until curl -s http://${QDRANT_HOST}:${QDRANT_PORT}/health > /dev/null 2>&1; do sleep 1; done"; then
    echo "❌ Qdrant is not accessible at ${QDRANT_HOST}:${QDRANT_PORT}"
    echo "💡 Tip: Make sure Qdrant container is running: docker-compose ps kolon-qdrant"
    exit 1
fi
echo "✅ Qdrant is accessible"

# 데이터 디렉토리 확인 (호스트 경로)
# 호스트에서 실행 시 상대 경로 사용, 컨테이너 내부에서는 절대 경로
HOST_DATA_DIR="${DATA_DIR:-../chatbot/car_data}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/backend/chatbot/car_data}"

# 호스트 경로 확인
if [ ! -d "$HOST_DATA_DIR" ]; then
    echo "❌ Data directory not found on host: $HOST_DATA_DIR"
    echo "💡 Tip: Make sure you're running from web-client directory"
    exit 1
fi
echo "📁 Host data directory: $HOST_DATA_DIR"
echo "📁 Container data directory: $CONTAINER_DATA_DIR"

# 데이터 로딩 실행
echo "📤 Loading data to Qdrant..."
CLEAR_EXISTING="${CLEAR_EXISTING:-true}"
docker-compose exec -T backend python -c "
from rag.data_loader import load_vehicle_data_to_qdrant
from pathlib import Path
import os

data_dir = Path('$CONTAINER_DATA_DIR')
clear_existing = '$CLEAR_EXISTING'.lower() == 'true'

print(f'📂 Using data directory: {data_dir}')
print(f'🗑️  Clear existing: {clear_existing}')

try:
    count = load_vehicle_data_to_qdrant(
        data_dir=data_dir,
        clear_existing=clear_existing
    )
    print(f'✅ Successfully loaded {count} points')
    exit(0)
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Qdrant initialization completed successfully"
else
    echo "❌ Qdrant initialization failed"
    exit 1
fi
