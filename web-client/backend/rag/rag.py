"""
RAG (Retrieval-Augmented Generation) pipeline for Interactive AI Dealer
Handles vector database operations and context retrieval using Qdrant
"""

import os
import re
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue

from openai import OpenAI
from dotenv import load_dotenv
from rag.vin_mapper import get_vin_mapper

load_dotenv()

ALL_SECTIONS = [
    "summary",
    "basic_info",
    "special_usage_history",
    "special_accident_history",
    "registration_change_history",
    "insurance_accident_history",
    "options",
    "seller_info",
    "inspection_record",
    "purpose_metadata",
]

EMBEDDING_DIMENSION_SIZE = 1536  # OpenAI text-embedding-3-small dimension


class RAGPipeline:
    """RAG pipeline for retrieving relevant context from Qdrant vector database"""

    def __init__(
        self,
        collection_name: str = "kolon_used_cars",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        openai_embed_model: str = "text-embedding-3-small",
    ):
        """
        Initialize RAG pipeline with Qdrant and OpenAI

        Args:
            collection_name: Name of the Qdrant collection
            qdrant_url: Qdrant server URL (for remote mode)
            qdrant_api_key: Qdrant API key (optional, for cloud Qdrant)
            qdrant_host: Qdrant server host (for local mode, default: localhost)
            qdrant_port: Qdrant server port (for local mode, default: 6333)
            openai_api_key: OpenAI API key (default: from environment)
            openai_embed_model: OpenAI embedding model name
        """
        self.collection_name = collection_name
        self.openai_embed_model = openai_embed_model

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        self.openai_client = OpenAI(api_key=api_key)

        # Initialize Qdrant client
        qdrant_kwargs = {}
        if qdrant_url:
            # 리모트 모드
            qdrant_kwargs["url"] = qdrant_url
            if qdrant_api_key:
                qdrant_kwargs["api_key"] = qdrant_api_key
        else:
            # 로컬 모드
            qdrant_kwargs["host"] = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
            qdrant_kwargs["port"] = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
            if qdrant_api_key:
                qdrant_kwargs["api_key"] = qdrant_api_key

        # Set timeout for large data operations (default: 60s, increased to 600s for bulk uploads)
        qdrant_timeout = float(os.getenv("QDRANT_TIMEOUT", "600.0"))
        qdrant_kwargs["timeout"] = qdrant_timeout

        self.client = QdrantClient(**qdrant_kwargs)

        # 컬렉션 존재 여부 확인 및 생성
        self._ensure_collection()

        # VIN mapper preload 제거: VIN은 basic_info 섹션에서 직접 추출하므로
        # 초기화 시점에 모든 JSON 파일을 로드할 필요가 없습니다.
        # VIN → car_id 변환이 필요한 경우에만 필요 시 로드됩니다.

    def _ensure_collection(self):
        """컬렉션 존재 여부 확인 및 생성"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name for col in collections.collections
            )

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"컬렉션 생성: {self.collection_name}")
        except Exception as e:
            print(f"⚠️  Error checking/creating collection: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """OpenAI API를 사용하여 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.openai_embed_model,
            input=text,
        )
        return response.data[0].embedding

    def search_raw_hits(
        self,
        query: str,
        sections: Optional[List[str]] = None,
        k_per_section: int = 10,
        query_year: Optional[int] = None,
        query_manufacturer: Optional[str] = None,
    ) -> List[ScoredPoint]:
        """
        Qdrant에서 관련 문서 검색 (청크 레벨)

        Args:
            query: 검색 쿼리 텍스트
            sections: 검색할 섹션 리스트 (None = 모든 섹션)
            k_per_section: 섹션별 결과 수 (기본값: 10)
            query_year: 추출된 연식 (LLM에서 추출한 값 사용)
            query_manufacturer: 추출된 제조사 (LLM에서 추출한 값 사용)

        Returns:
            ScoredPoint 객체 리스트
        """
        query_vector = self.get_embedding(query)
        hits: List[ScoredPoint] = []

        target_sections = sections or ALL_SECTIONS

        for section in target_sections:
            filter_conditions = [
                FieldCondition(
                    key="section",
                    match=MatchValue(value=section),
                )
            ]

            # basic_info 섹션에서만 연식 필터링 가능
            if section == "basic_info" and query_year:
                filter_conditions.append(
                    FieldCondition(
                        key="raw.year",
                        match=MatchValue(value=query_year),
                    )
                )

            # 제조사 필터링: basic_info 섹션에만 manufacturer 필드가 있으므로 basic_info에서만 필터링
            # 다른 섹션에서는 필터링하지 않음 (벡터 검색 점수에 의존)
            # 평탄화된 manufacturer 또는 raw.manufacturer 둘 다 확인
            if section == "basic_info" and query_manufacturer:
                # 평탄화된 manufacturer 필드 사용 (data_loader에서 raw.manufacturer를 payload.manufacturer로 복사)
                filter_conditions.append(
                    FieldCondition(
                        key="manufacturer",
                        match=MatchValue(value=query_manufacturer),
                    )
                )

            section_filter = Filter(must=filter_conditions)

            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=k_per_section,
                    query_filter=section_filter,
                )
                hits.extend(response.points)
            except Exception as e:
                print(f"⚠️  Error searching section '{section}': {e}")
        return hits

    def aggregate_by_car(
        self,
        hits: List[ScoredPoint],
        section_weights: Optional[Dict[str, float]] = None,
        query: Optional[str] = None,
        query_year: Optional[int] = None,
        query_model_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        차량 ID별로 검색 결과 집계 (차량 레벨 순위 생성)

        Args:
            hits: 검색 결과로부터 얻은 ScoredPoint 리스트
            section_weights: 섹션별 가중치 (선택적)
            query: 원본 검색 쿼리 (재랭킹용, 선택적)

        Returns:
            car_id를 키로 하는 딕셔너리, 집계된 점수와 청크 리스트를 값으로 함
        """
        import re

        # 기본 섹션 가중치 설정
        default_weights = {
            "basic_info": 3.0,  # basic_info에 높은 가중치
            "summary": 2.0,
            "options": 1.5,
            "purpose_metadata": 1.2,
        }

        if section_weights is None:
            section_weights = default_weights
        else:
            # 기본 가중치와 병합
            for section, weight in default_weights.items():
                if section not in section_weights:
                    section_weights[section] = weight

        # query_year와 query_model_keywords가 파라미터로 전달되지 않았으면 쿼리에서 추출 시도 (fallback)
        if query_year is None and query:
            year_match = re.search(r"(\d{4})년?식?", query)
            if year_match:
                try:
                    query_year = int(year_match.group(1))
                except ValueError:
                    pass

        if query_model_keywords is None:
            query_model_keywords = []

        grouped = defaultdict(lambda: {"score": 0.0, "chunks": []})

        for hit in hits:
            payload = hit.payload or {}
            car_id = payload.get("car_id")
            if not car_id:
                continue

            weight = 1.0
            section = payload.get("section")
            if section_weights:
                weight = section_weights.get(section, 1.0)

            # basic_info 섹션에서 연식/모델명 매칭 확인
            # 쿼리에 명시적인 정보가 있을 때만 보너스/페널티 적용
            if section == "basic_info" and query:
                raw_data = payload.get("raw", {})

                # 연식 매칭 보너스 (연식이 쿼리에 명시된 경우에만)
                if query_year:
                    if raw_data.get("year") == query_year:
                        weight *= 2.0  # 연식이 정확히 일치하면 2배 가중치
                    else:
                        # 연식이 다르면 페널티 (하지만 완전히 제거하지는 않음)
                        year_diff = abs(raw_data.get("year", 0) - query_year)
                        if year_diff > 2:  # 2년 이상 차이나면 페널티
                            weight *= 0.3

                # 모델명 키워드 매칭 보너스 (키워드가 쿼리에 명시된 경우에만)
                if query_model_keywords:
                    model_name = raw_data.get("model_name", "").lower()
                    matched_keywords = sum(
                        1
                        for keyword in query_model_keywords
                        if keyword.lower() in model_name
                    )
                    if matched_keywords > 0:
                        weight *= 1.0 + (matched_keywords * 0.5)  # 키워드당 50% 보너스
                    else:
                        # 모델명 키워드가 전혀 매칭되지 않으면 페널티
                        weight *= 0.5

            grouped[car_id]["score"] += float(hit.score) * weight
            grouped[car_id]["chunks"].append(hit)

        return dict(grouped)

    def select_top_cars(
        self,
        grouped: Dict[str, Dict[str, Any]],
        top_n: int = 3,
    ) -> List[tuple]:
        """Select top N cars by aggregated score"""
        cars = sorted(grouped.items(), key=lambda x: x[1]["score"], reverse=True)

        # 디버깅: 상위 차량 점수 로깅
        if cars:
            print(f"[RAG] 상위 {min(top_n, len(cars))}개 차량 점수:")
            for i, (car_id, car_data) in enumerate(cars[:top_n], 1):
                score = car_data["score"]
                chunks_count = len(car_data["chunks"])
                print(f"  {i}. {car_id}: 점수={score:.4f}, 청크수={chunks_count}")

        return cars[:top_n]

    def _get_vin_from_basic_info(self, car_id: str) -> Optional[str]:
        """
        car_id의 basic_info 섹션만 조회하여 VIN 추출
        VIN은 basic_info 섹션의 payload.raw.vin에 저장되어 있습니다.

        Args:
            car_id: 차량 ID

        Returns:
            VIN 번호 또는 None
        """
        try:
            vehicle_filter = Filter(
                must=[
                    FieldCondition(key="car_id", match=MatchValue(value=car_id)),
                    FieldCondition(key="section", match=MatchValue(value="basic_info")),
                ]
            )

            dummy_vector = [0.0] * EMBEDDING_DIMENSION_SIZE
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dummy_vector,
                limit=1,
                query_filter=vehicle_filter,
            )

            if response.points:
                payload = response.points[0].payload or {}
                raw_data = payload.get("raw", {})
                if isinstance(raw_data, dict) and "vin" in raw_data:
                    return raw_data["vin"]
        except Exception as e:
            print(f"⚠️  Error retrieving VIN for car_id '{car_id}': {e}")

        return None

    def search(
        self,
        query: str,
        n_results: int = 5,
        sections: Optional[List[str]] = None,
        filter_metadata: Optional[Dict] = None,
        extraction_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Search for relevant documents (improved: aggregates by car and returns top cars)

        Args:
            query: Search query text
            n_results: Number of cars to return (not chunks)
            sections: Optional list of sections to search
            filter_metadata: Optional metadata filter (deprecated, use sections instead)
            extraction_info: 구조화된 추출 정보 (year, manufacturer, model_keywords) - LLM에서 추출

        Returns:
            List of dictionaries with 'document', 'metadata', 'distance', 'id'
            Now returns top cars with aggregated information from all their sections
        """
        # extraction_info에서 재랭킹용 정보 추출
        # extraction_info가 없을 때를 대비해 기본값 설정
        query_year = None
        query_manufacturer = None
        query_model_keywords = []

        if extraction_info:
            query_year = extraction_info.get("year")
            query_manufacturer = extraction_info.get("manufacturer")
            query_model_keywords = extraction_info.get("model_keywords", [])

        hits = self.search_raw_hits(
            query,
            sections=sections,
            k_per_section=10,
            query_year=query_year,
            query_manufacturer=query_manufacturer,
        )

        # Aggregate by car_id with query-aware weighting
        grouped = self.aggregate_by_car(
            hits,
            query=query,
            query_year=query_year,
            query_model_keywords=query_model_keywords,
        )

        # Select top N cars
        top_cars = self.select_top_cars(grouped, top_n=n_results)

        # Format results: for each car, combine sections into a single document
        formatted_results = []
        for car_id, car_data in top_cars:
            chunks = car_data["chunks"]
            aggregated_score = car_data["score"]

            # Combine all sections from this car into a single document
            # Sort chunks by section order for consistent output
            section_order = {section: idx for idx, section in enumerate(ALL_SECTIONS)}
            chunks.sort(
                key=lambda c: section_order.get(c.payload.get("section", ""), 999)
            )

            # Collect all nl texts from chunks
            car_documents = []
            for chunk in chunks:
                payload = chunk.payload or {}
                nl = payload.get("nl", "")
                if nl:
                    section = payload.get("section", "unknown")
                    car_documents.append(f"[{section}] {nl}")

            combined_document = "\n".join(car_documents)

            # Use metadata from first chunk (car_id and section will be consistent)
            if chunks:
                first_payload = chunks[0].payload or {}
                metadata = {
                    k: v for k, v in first_payload.items() if k not in ["nl", "raw"]
                }
                metadata["car_id"] = car_id
                # Add aggregated score
                metadata["aggregated_score"] = aggregated_score

                # VIN 추출 최적화: chunks에서 basic_info 섹션 찾기
                # VIN은 basic_info 섹션의 payload.raw.vin에 저장되어 있습니다
                vin = None
                for chunk in chunks:
                    payload = chunk.payload or {}
                    if payload.get("section") == "basic_info":
                        raw_data = payload.get("raw", {})
                        if isinstance(raw_data, dict) and "vin" in raw_data:
                            vin = raw_data["vin"]
                            break

                # chunks에 basic_info가 없으면 basic_info 섹션만 조회
                if not vin:
                    vin = self._get_vin_from_basic_info(car_id)

                if vin:
                    metadata["vin"] = vin

                formatted_results.append(
                    {
                        "document": combined_document,
                        "metadata": metadata,
                        "distance": (
                            1.0 - (aggregated_score / len(chunks)) if chunks else 1.0
                        ),  # Average score
                        "id": f"{car_id}_aggregated",
                        "score": (
                            aggregated_score / len(chunks) if chunks else 0.0
                        ),  # Average score
                    }
                )

        return formatted_results

    def get_context(
        self,
        query: str,
        n_results: int = 3,
        sections: Optional[List[str]] = None,
        extraction_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get formatted context string from search results (improved: groups by car)

        Args:
            query: Search query
            n_results: Number of cars to include
            sections: Optional list of sections to search
            extraction_info: 구조화된 추출 정보 (year, manufacturer, model_keywords) - LLM에서 추출

        Returns:
            Formatted context string grouped by car with all sections combined
        """
        results = self.search(
            query,
            n_results=n_results,
            sections=sections,
            extraction_info=extraction_info,
        )

        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result["document"]
            metadata = result.get("metadata", {})
            car_id = metadata.get("car_id", "unknown")
            score = result.get("score", 0)

            context_parts.append(f"[{i}] Car: {car_id} (Score: {score:.4f})\n{doc}")

        return "\n\n".join(context_parts)

    def get_vehicle_full_info(
        self,
        vehicle_id: str,
        sections: Optional[List[str]] = None,
    ) -> str:
        """
        Get complete information for a specific vehicle by vehicle_id
        Retrieves all sections (or specified sections) for the vehicle in a logical order

        Args:
            vehicle_id: Vehicle ID to retrieve (VIN or car_id, e.g., "WBAU6D6U6R3R15LMG" or "볼보_2024_001")
            sections: Optional list of sections to retrieve (None = all sections)

        Returns:
            Formatted string with all (or specified) sections of the vehicle information
        """
        # VIN인 경우에만 VIN 매퍼 사용 (VIN → car_id 변환)
        # car_id인 경우는 그대로 사용하고, VIN은 나중에 basic_info에서 추출
        vin_mapper = get_vin_mapper()
        if vin_mapper.is_vin(vehicle_id):
            # VIN인 경우: car_id로 변환 필요
            car_id = vin_mapper.get_car_id_from_vin(vehicle_id)
            if not car_id:
                return f"No information found for vehicle VIN: {vehicle_id}"
            search_id = car_id
            vin = vehicle_id
        else:
            # car_id인 경우: 그대로 사용, VIN은 나중에 basic_info에서 추출
            search_id = vehicle_id
            vin = None

        # Define section order for logical presentation
        all_sections_order = [
            "summary",
            "basic_info",
            "purpose_metadata",
            "options",
            "special_usage_history",
            "special_accident_history",
            "registration_change_history",
            "insurance_accident_history",
            "inspection_record",
            "seller_info",
        ]

        # Filter sections if specified
        sections_order = sections if sections else all_sections_order
        # Maintain order from all_sections_order
        sections_order = [s for s in all_sections_order if s in sections_order]

        context_parts = []
        # VIN이 있으면 표시, 없으면 vehicle_id 표시
        # car_id인 경우 basic_info에서 VIN 추출 시도
        if not vin:
            vin = self._get_vin_from_basic_info(search_id)
        display_id = vin if vin else vehicle_id
        context_parts.append(f"=== Vehicle: {display_id} ===\n")

        # Query each section for this vehicle
        for section in sections_order:
            try:
                # Create filter for this vehicle_id and section
                # search_id는 이미 car_id로 변환됨
                vehicle_filter = Filter(
                    must=[
                        FieldCondition(key="car_id", match=MatchValue(value=search_id)),
                        FieldCondition(key="section", match=MatchValue(value=section)),
                    ]
                )

                # Query Qdrant (we don't need embedding for exact match, but we need to use query_points)
                # For exact vehicle_id match, we can use a dummy query vector since filter will handle matching
                dummy_vector = [0.0] * EMBEDDING_DIMENSION_SIZE
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dummy_vector,
                    limit=1,  # Should only be one record per car_id + section
                    query_filter=vehicle_filter,
                )

                if response.points:
                    point = response.points[0]
                    payload = point.payload or {}
                    nl = payload.get("nl", "")

                    if nl:
                        context_parts.append(
                            f"--- {section.upper().replace('_', ' ')} ---"
                        )
                        context_parts.append(nl)
                        context_parts.append("")  # Empty line for readability

            except Exception as e:
                print(
                    f"⚠️  Error retrieving section '{section}' for vehicle '{search_id}': {e}"
                )

        if len(context_parts) <= 2:  # Only header and maybe one section
            # VIN이 아직 없으면 다시 시도
            if not vin:
                vin = self._get_vin_from_basic_info(search_id)
            display_id = vin if vin else vehicle_id
            return f"No information found for vehicle: {display_id}"

        return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton"""
    global _rag_pipeline

    if _rag_pipeline is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "kolon_used_cars")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_host = os.getenv("QDRANT_HOST")
        qdrant_port = os.getenv("QDRANT_PORT")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        _rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=int(qdrant_port) if qdrant_port else None,
            openai_api_key=openai_api_key,
            openai_embed_model=openai_embed_model,
        )
    return _rag_pipeline
