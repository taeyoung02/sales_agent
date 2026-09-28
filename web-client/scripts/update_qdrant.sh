#!/bin/bash
# Qdrant 데이터 업데이트 스크립트 (기존 데이터 유지하며 업데이트)

set -e

echo "🔄 Updating Qdrant data (incremental update)..."

DATA_DIR="${DATA_DIR:-/backend/chatbot/car_data}"
CLEAR_EXISTING="${CLEAR_EXISTING:-false}"

docker-compose exec -T backend python -c "
from rag.data_loader import load_vehicle_data_to_qdrant
from pathlib import Path
import os

data_dir = Path(os.getenv('DATA_DIR', '$DATA_DIR'))
clear_existing = os.getenv('CLEAR_EXISTING', '$CLEAR_EXISTING').lower() == 'true'

try:
    count = load_vehicle_data_to_qdrant(
        data_dir=data_dir,
        clear_existing=clear_existing  # false면 기존 데이터 유지
    )
    print(f'✅ Successfully updated {count} points')
    exit(0)
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
