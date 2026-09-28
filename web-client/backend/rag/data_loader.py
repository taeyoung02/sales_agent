"""
Vehicle Data Loader
Loads car_info*_vector.json files from web-client/source/car_data directory into Qdrant
"""

import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional

# Add backend directory to Python path for imports
# This ensures imports work when running as a script from any location
_backend_dir = Path(__file__).parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from qdrant_client.http.models import PointStruct

from rag.rag import get_rag_pipeline
from utils.path_utils import get_project_root


def load_vector_files(data_dir: Path) -> List[dict]:
    """
    Load all *_vector.json files from data directory

    Args:
        data_dir: Directory containing vector JSON files

    Returns:
        List of all records from all vector files
    """
    all_records = []
    # 모든 *_vector.json 파일 찾기 (car_info*_vector.json 포함)
    vector_files = sorted(data_dir.glob("*_vector.json"))

    if not vector_files:
        raise FileNotFoundError(f"No *_vector.json files found in {data_dir}")

    for vector_file in vector_files:
        print(f"📄 Loading: {vector_file.name}")
        try:
            records = json.loads(vector_file.read_text(encoding="utf-8"))
            if isinstance(records, list):
                all_records.extend(records)
            else:
                print(f"⚠️  Skipping {vector_file.name}: not a list")
        except Exception as e:
            print(f"⚠️  Error loading {vector_file.name}: {e}")

    return all_records


def load_vehicle_data_to_qdrant(
    data_dir: Optional[Path] = None,
    clear_existing: bool = False,
) -> int:
    """
    Load vehicle vector data from JSON files into Qdrant

    Args:
        data_dir: Directory containing car_info*_vector.json files
                  (default: project_root/web-client/source/car_data)
        clear_existing: Whether to clear existing collection before loading

    Returns:
        Number of points loaded
    """
    # Determine data directory
    if data_dir is None:
        # Assume we're in web-client/backend, go up to project root
        current_file = Path(__file__)
        data_dir = get_project_root() / "web-client" / "source" / "new_car_data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"📁 Data directory: {data_dir}")

    # Load all vector files
    records = load_vector_files(data_dir)
    print(f"📊 Loaded {len(records)} records from vector files")

    if not records:
        raise ValueError("No records found in vector files")

    # Get RAG pipeline
    rag = get_rag_pipeline()

    # Clear existing collection if requested
    if clear_existing:
        try:
            collections = rag.client.get_collections()
            collection_exists = any(
                col.name == rag.collection_name for col in collections.collections
            )
            if collection_exists:
                rag.client.delete_collection(collection_name=rag.collection_name)
                print(f"🗑️  Deleted existing collection: {rag.collection_name}")

            # Recreate collection
            from qdrant_client.models import VectorParams, Distance

            rag.client.create_collection(
                collection_name=rag.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created new collection: {rag.collection_name}")
        except Exception as e:
            print(f"⚠️  Error managing collection: {e}")

    # Convert records to Qdrant points
    points = []
    for rec in records:
        # Generate point ID from record ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rec["id"]))
        
        # Payload 복사 및 메타데이터 평탄화 (manufacturer, model_name을 최상위 레벨에 추가)
        payload = rec["payload"].copy() if isinstance(rec["payload"], dict) else rec["payload"]
        
        # basic_info 섹션에서 manufacturer, model_name 추출하여 평탄화
        if isinstance(payload, dict):
            raw_data = payload.get("raw", {})
            if isinstance(raw_data, dict):
                # manufacturer 평탄화 (필터링을 위해)
                if "manufacturer" in raw_data and "manufacturer" not in payload:
                    payload["manufacturer"] = raw_data["manufacturer"]
                # model_name 평탄화 (필터링을 위해)
                if "model_name" in raw_data and "model_name" not in payload:
                    payload["model_name"] = raw_data["model_name"]

        points.append(
            PointStruct(
                id=point_id,
                vector=rec["vector"],
                payload=payload,
            )
        )

    # Upsert to Qdrant
    print(f"📤 Uploading {len(points)} points to Qdrant...")
    rag.client.upsert(
        collection_name=rag.collection_name,
        points=points,
    )
    print(f"✅ Successfully uploaded {len(points)} points")

    # Verify count
    total = rag.client.count(collection_name=rag.collection_name, exact=True)
    print(f"📊 Total points in collection: {total.count}")

    return len(points)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Load vehicle data to Qdrant vector database"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing car_info*_vector.json files (default: project_root/web-client/source/car_data)",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing collection before loading",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    try:
        count = load_vehicle_data_to_qdrant(
            data_dir=data_dir,
            clear_existing=args.clear_existing,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
