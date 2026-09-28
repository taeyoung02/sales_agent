"""
VIN (Vehicle Identification Number) 매핑 유틸리티
VIN과 car_id 간의 매핑을 관리합니다.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.path_utils import get_source_root


class VINMapper:
    """VIN과 car_id 간의 매핑을 관리하는 클래스"""

    def __init__(self):
        self._vin_to_car_id: Dict[str, str] = {}
        self._car_id_to_vin: Dict[str, str] = {}
        self._loaded = False

    def _load_single_file(self, json_file: Path) -> Dict[str, tuple]:
        """
        단일 JSON 파일에서 VIN → car_id 매핑 추출 (병렬 처리용)

        Returns:
            Dict with keys: 'mappings' (list of (vin, car_id) tuples), 'errors' (list of error messages)
        """
        mappings = []
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 데이터가 리스트인 경우 (vector JSON 형식)
            if isinstance(data, list):
                for record in data:
                    if not isinstance(record, dict):
                        continue

                    payload = record.get("payload", {})
                    if not isinstance(payload, dict):
                        continue

                    raw = payload.get("raw", {})
                    car_id = payload.get("car_id")

                    # raw에서 VIN 추출
                    vin = None
                    if isinstance(raw, dict):
                        vin = raw.get("vin")

                    if car_id and vin:
                        mappings.append((vin, car_id))
            # 데이터가 딕셔너리인 경우 (단일 차량 JSON)
            elif isinstance(data, dict):
                raw = data.get("raw", {})
                car_id = data.get("car_id")
                vin = None
                if isinstance(raw, dict):
                    vin = raw.get("vin")
                if car_id and vin:
                    mappings.append((vin, car_id))

        except Exception as e:
            return {"mappings": [], "error": f"Error loading {json_file.name}: {e}"}

        return {"mappings": mappings, "error": None}

    def _load_mapping(self, data_dir: Optional[Path] = None):
        """JSON 파일에서 VIN → car_id 매핑 로드 (병렬 처리로 최적화)"""
        if self._loaded:
            return

        if data_dir is None:
            # 기본 경로: new_car_data 또는 car_data
            # Docker 환경에서는 /backend/source, 로컬에서는 web-client/source
            source_root = get_source_root()
            new_car_data_dir = source_root / "new_car_data"
            car_data_dir = source_root / "car_data"

            # new_car_data 우선, 없으면 car_data 사용
            if new_car_data_dir.exists():
                data_dir = new_car_data_dir
            elif car_data_dir.exists():
                data_dir = car_data_dir
            else:
                print(
                    f"⚠️  Data directory not found: {new_car_data_dir} or {car_data_dir}"
                )
                self._loaded = True
                return

        if not data_dir.exists():
            print(f"⚠️  Data directory not found: {data_dir}")
            self._loaded = True
            return

        # JSON 파일 찾기 (*.json, *_vector.json 제외)
        json_files = [
            f for f in data_dir.glob("*.json") if not f.name.endswith("_vector.json")
        ]

        if not json_files:
            print("⚠️  No JSON files found")
            self._loaded = True
            return

        # 병렬 처리로 파일 로딩 (최대 8개 워커 스레드)
        total_mappings = 0
        errors = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 모든 파일 로딩 작업 제출
            future_to_file = {
                executor.submit(self._load_single_file, json_file): json_file
                for json_file in json_files
            }

            # 결과 수집
            for future in as_completed(future_to_file):
                result = future.result()
                if result.get("error"):
                    errors.append(result["error"])
                else:
                    mappings = result.get("mappings", [])
                    for vin, car_id in mappings:
                        self._vin_to_car_id[vin] = car_id
                        self._car_id_to_vin[car_id] = vin
                        total_mappings += 1

        if errors:
            print(f"⚠️  {len(errors)} files had errors during loading")
            if len(errors) <= 5:  # 에러가 적으면 모두 출력
                for error in errors:
                    print(f"  - {error}")

        print(f"✅ Loaded {total_mappings} VIN mappings from {len(json_files)} files")
        self._loaded = True

    def get_car_id_from_vin(
        self, vin: str, data_dir: Optional[Path] = None
    ) -> Optional[str]:
        """VIN으로부터 car_id 조회 (최적화: 로드 체크 최소화)"""
        if not self._loaded:
            self._load_mapping(data_dir)
        # 이미 로드된 경우 바로 딕셔너리 조회만 수행 (O(1))
        return self._vin_to_car_id.get(vin)

    def get_vin_from_car_id(
        self, car_id: str, data_dir: Optional[Path] = None
    ) -> Optional[str]:
        """car_id로부터 VIN 조회 (최적화: 로드 체크 최소화)"""
        if not self._loaded:
            self._load_mapping(data_dir)
        # 이미 로드된 경우 바로 딕셔너리 조회만 수행 (O(1))
        return self._car_id_to_vin.get(car_id)

    def get_vins_from_car_ids(
        self, car_ids: List[str], data_dir: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """
        여러 car_id에 대한 VIN을 배치로 조회 (최적화)

        Args:
            car_ids: 조회할 car_id 리스트
            data_dir: 데이터 디렉토리 (선택적)

        Returns:
            Dict[car_id, vin] - 각 car_id에 대한 VIN 매핑
        """
        if not self._loaded:
            self._load_mapping(data_dir)

        # 배치 조회로 성능 최적화
        return {car_id: self._car_id_to_vin.get(car_id) for car_id in car_ids}

    def preload(self, data_dir: Optional[Path] = None) -> None:
        """매핑을 미리 로드 (초기화 시점에 호출)"""
        if not self._loaded:
            self._load_mapping(data_dir)

    def is_vin(self, vehicle_id: str) -> bool:
        """vehicle_id가 VIN 형식인지 확인 (17자리 알파벳+숫자)"""
        if not vehicle_id:
            return False
        # VIN은 보통 17자리 알파벳과 숫자 조합
        return len(vehicle_id) == 17 and vehicle_id.isalnum()

    def normalize_vehicle_id(
        self, vehicle_id: str, data_dir: Optional[Path] = None
    ) -> tuple[str, Optional[str]]:
        """
        vehicle_id를 정규화하여 (car_id, vin) 튜플 반환 (최적화: 로드 체크 최소화)

        Returns:
            (car_id, vin) 튜플
            - vehicle_id가 VIN이면: (car_id, vin)
            - vehicle_id가 car_id이면: (car_id, vin)
            - 매핑을 찾을 수 없으면: (vehicle_id, None)
        """
        if not self._loaded:
            self._load_mapping(data_dir)

        if not vehicle_id:
            return (vehicle_id, None)

        # VIN 형식인지 확인 (빠른 체크)
        if self.is_vin(vehicle_id):
            # 이미 로드된 경우 바로 딕셔너리 조회만 수행
            car_id = self._vin_to_car_id.get(vehicle_id)
            return (car_id or vehicle_id, vehicle_id)

        # car_id 형식인지 확인 (이미 로드된 경우 바로 딕셔너리 조회)
        vin = self._car_id_to_vin.get(vehicle_id)
        if vin:
            return (vehicle_id, vin)

        # 매핑을 찾을 수 없으면 그대로 반환
        return (vehicle_id, None)


# 싱글톤 인스턴스
_vin_mapper: Optional[VINMapper] = None


def get_vin_mapper() -> VINMapper:
    """VIN 매퍼 싱글톤 인스턴스 반환"""
    global _vin_mapper
    if _vin_mapper is None:
        _vin_mapper = VINMapper()
    return _vin_mapper
