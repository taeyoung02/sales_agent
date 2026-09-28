#!/bin/bash
# Qdrant 상태 확인 스크립트

set -e

QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION="${QDRANT_COLLECTION:-kolon_used_cars}"

echo "🔍 Checking Qdrant status..."

# Qdrant 헬스체크
echo "📊 Qdrant Health:"
curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/health" | jq '.' || echo "⚠️  Health check failed"

# 컬렉션 정보
echo ""
echo "📚 Collection Info:"
docker-compose exec -T backend python -c "
from rag.rag import get_rag_pipeline

rag = get_rag_pipeline()
client = rag.client
collection = rag.collection_name

try:
    info = client.get_collection(collection_name=collection)
    count = client.count(collection_name=collection, exact=True)
    print(f'Collection: {collection}')
    print(f'Points count: {count.count}')
    print(f'Vectors count: {info.config.params.vectors.size}')
    print(f'Status: {info.status}')
except Exception as e:
    print(f'❌ Error: {e}')
"
