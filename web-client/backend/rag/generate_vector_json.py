"""
Generate vector JSON files from car_info JSON files
Converts car_info*.json to car_info*_vector.json by adding embeddings
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use web-client/source/car_data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "web-client" / "source" / "new_car_data"


def iter_cardata_files(directory: Path) -> Iterable[Path]:
    """car_info*.json 중 *_vector.json 파일은 제외하고 순회."""
    for path in sorted(directory.glob("*.json")):
        if path.stem.endswith("_vector"):
            continue
        yield path


def embed_file(path: Path) -> None:
    """JSON 파일의 각 항목에 대해 embedding을 생성하고 vector 필드를 채움"""
    print(f"\nProcessing {path.name}")
    try:
        cardata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  ✗ JSON decode failed: {exc}. Skipping this file.")
        return

    if not isinstance(cardata, list):
        print(f"  ✗ Expected list, got {type(cardata)}. Skipping this file.")
        return

    for idx, item in enumerate(cardata):
        # payload.nl 필드에서 텍스트 추출
        text = item.get("payload", {}).get("nl", "")
        if not text:
            print(f"  ⚠️  Item {idx + 1} has no 'nl' field, skipping embedding")
            continue

        # OpenAI API로 embedding 생성
        try:
            resp = client.embeddings.create(
                model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                input=text,
            )
            item["vector"] = resp.data[0].embedding
            print(f"  → Processed {idx + 1} / {len(cardata)}")
        except Exception as e:
            print(f"  ✗ Error generating embedding for item {idx + 1}: {e}")
            continue

    # _vector.json 파일로 저장
    output = path.with_name(f"{path.stem}_vector.json")
    output.write_text(
        json.dumps(cardata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ Saved {output.name}")


def main() -> None:
    """메인 함수: DATA_DIR의 모든 car_info*.json 파일을 처리"""
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        print(f"   Please ensure the directory exists.")
        return

    print(f"📁 Data directory: {DATA_DIR}")

    files = list(iter_cardata_files(DATA_DIR))
    if not files:
        print(f"⚠️  No car_info*.json files found in {DATA_DIR}")
        return

    print(f"📄 Found {len(files)} file(s) to process\n")

    for file_path in files:
        embed_file(file_path)

    print(f"\n✅ All files processed!")


if __name__ == "__main__":
    main()
