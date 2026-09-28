# Router: 컨텍스트 수집 전용 (리드 분류 없음)
# 유저 정보 추출과 RAG 검색을 병렬로 수행하고 Planner에 넘긴다.

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from prompts import PROMPT_REGISTRY
from rag.rag import ALL_SECTIONS
from utils.logging_context import set_session_id

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

_GREETING_RE = re.compile(
    r"^(안녕(하세요|하십니까)?|하이|헬로|hi+|hello|hey|반가워|반갑습니다)"
    r"[\s!.~ㅎㅋ]*$",
    re.IGNORECASE,
)


class SearchQueryExtraction(BaseModel):
    """RAG 검색을 위한 쿼리 및 섹션 추출 결과"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    sections: List[str] = Field(
        description="필요한 차량 정보 섹션 리스트 (예: ['summary', 'basic_info', 'options'])",
    )
    search_query: str = Field(
        description="RAG 검색에 최적화된 쿼리",
    )
    year: Optional[int] = Field(default=None, description="쿼리에서 추출한 연식")
    manufacturer: Optional[str] = Field(
        default=None, description="쿼리에서 추출한 제조사"
    )
    model_keywords: List[str] = Field(
        default_factory=list,
        description="쿼리에서 추출한 모델명 관련 키워드",
    )


class LLMRouter(BaseAgent):
    """컨텍스트 수집 Router. 리드 분류는 Planner Step1에서 수행한다."""

    def collect(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        vehicle_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        유저 정보와 RAG를 병렬로 모은다.

        Returns:
            rag_context, user_info, guess_sections, search_query, evidence
        """
        set_session_id(session_id or "default")
        skip_rag = self._should_skip_rag(message, vehicle_id)

        logger.info("=" * 80)
        logger.info("[Router] 컨텍스트 수집 시작")
        logger.info(f"  - skip_rag={skip_rag}, vehicle_id={vehicle_id}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_user = pool.submit(
                self._extract_user_info, message, conversation_history, session_id
            )
            if skip_rag:
                rag_payload = {
                    "rag_context": None,
                    "guess_sections": None,
                    "search_query": message,
                    "evidence": [],
                }
                user_info = future_user.result()
            else:
                future_rag = pool.submit(
                    self._retrieve,
                    message,
                    vehicle_id,
                    conversation_history,
                    session_id,
                )
                user_info = future_user.result()
                rag_payload = future_rag.result()

        user_info = user_info or self.get_user_info(session_id)
        rag_context = rag_payload.get("rag_context")
        logger.info(
            "[Router] 수집 완료: user_info=%s, rag_len=%s",
            bool(user_info),
            len(rag_context) if rag_context else 0,
        )
        logger.info("=" * 80)

        return {
            "rag_context": rag_context,
            "user_info": user_info,
            "guess_sections": rag_payload.get("guess_sections"),
            "search_query": rag_payload.get("search_query") or message,
            "evidence": rag_payload.get("evidence") or [],
        }

    def _should_skip_rag(self, message: str, vehicle_id: Optional[str]) -> bool:
        if vehicle_id:
            return False
        text = (message or "").strip()
        if not text:
            return True
        return bool(_GREETING_RE.match(text))

    def _retrieve(
        self,
        message: str,
        vehicle_id: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        guess_sections, search_query, extraction_info = self._extract_guess_sections(
            message, conversation_history, session_id
        )
        rag_context = self._perform_rag_search(
            search_query,
            vehicle_id,
            conversation_history,
            guess_sections,
            extraction_info,
        )
        evidence = self._build_evidence(search_query, rag_context)
        return {
            "rag_context": rag_context,
            "guess_sections": guess_sections,
            "search_query": search_query,
            "evidence": evidence,
        }

    def _extract_user_info(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        set_session_id(session_id or "default")
        try:
            prompt_config = PROMPT_REGISTRY.get("user_info_update", {})
            system_prompt = prompt_config.get("system", "")
            user_template = prompt_config.get("user_template", "")
            current_info = self.get_user_info(session_id)
            current_info_json = json.dumps(current_info, ensure_ascii=False)
            user_prompt = user_template.format(
                customer_info=current_info_json, question=message
            )
            output_text = self.llm_client.generate_text(
                messages=[{"role": "user", "content": user_prompt}],
                system_instruction=system_prompt,
                use_fast_model=True,
                session_id=session_id,
            )
            output_text = (output_text or "").strip()
            if output_text.startswith("```"):
                output_text = output_text.split("```")[1]
                if output_text.startswith("json"):
                    output_text = output_text[4:]
                output_text = output_text.strip()
            try:
                extracted_info = json.loads(output_text)
            except json.JSONDecodeError:
                logger.warning("[Router] UserInfo JSON 파싱 실패: %s", output_text)
                extracted_info = {}
            updated_info = self.update_user_info(extracted_info, session_id)
            logger.info(
                "[Router] 사용자 정보: %s", json.dumps(updated_info, ensure_ascii=False)
            )
            return updated_info
        except Exception as e:
            logger.error("[Router] UserInfo 오류: %s", e, exc_info=True)
            return self.get_user_info(session_id)

    def _extract_guess_sections(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[List[str]], str, Optional[Dict[str, Any]]]:
        try:
            history_summary = ""
            if conversation_history:
                effective_history = self.get_effective_history(
                    conversation_history, session_id=session_id, recent_count=5
                )
                history_summary = "\n".join(
                    [
                        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                        for msg in effective_history
                    ]
                )
            prompt_config = PROMPT_REGISTRY.get("extract_search_query_and_sections", {})
            system_prompt = prompt_config.get("system", "")
            user_template = prompt_config.get("user_template", "")
            user_prompt = user_template.format(
                question=message, history_summary=history_summary or "(대화 없음)"
            )
            extraction_result = self.llm_client.generate_structured(
                messages=[{"role": "user", "content": user_prompt}],
                response_schema=SearchQueryExtraction,
                system_instruction=system_prompt,
                use_fast_model=True,
                session_id=session_id,
            )
            filtered_sections = [
                s for s in extraction_result.sections if s in ALL_SECTIONS
            ]
            sections = filtered_sections if filtered_sections else ALL_SECTIONS
            search_query = extraction_result.search_query.strip() or message
            extraction_info = {
                "year": extraction_result.year,
                "manufacturer": extraction_result.manufacturer,
                "model_keywords": extraction_result.model_keywords,
            }
            return sections, search_query, extraction_info
        except Exception as e:
            logger.error("[Router] GuessSections 오류: %s", e, exc_info=True)
            return ALL_SECTIONS, message, None

    def _perform_rag_search(
        self,
        message: str,
        vehicle_id: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        guess_sections: Optional[List[str]] = None,
        extraction_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """VIN 정확 조회 → 섹션 필터 검색 → 필터 해제 → 실패 시 None."""
        if guess_sections:
            logger.info("[Router] RAG 섹션 필터: %s", guess_sections)

        target_vehicle_id = vehicle_id
        if not target_vehicle_id and conversation_history:
            target_vehicle_id = self.extract_vehicle_id_from_history(
                conversation_history
            )

        if target_vehicle_id:
            vehicle_info = self.rag_pipeline.get_vehicle_full_info(
                target_vehicle_id, sections=guess_sections
            )
            if vehicle_info and not vehicle_info.startswith("No information found"):
                return vehicle_info
            if guess_sections:
                vehicle_info_fallback = self.rag_pipeline.get_vehicle_full_info(
                    target_vehicle_id, sections=None
                )
                if vehicle_info_fallback and not vehicle_info_fallback.startswith(
                    "No information found"
                ):
                    return vehicle_info_fallback

        return self._perform_general_search_with_fallback(
            message, guess_sections, extraction_info
        )

    def _perform_general_search_with_fallback(
        self,
        message: str,
        guess_sections: Optional[List[str]],
        extraction_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        logger.info("[Router] RAG 일반 검색: %s", message)
        try:
            context = self.rag_pipeline.get_context(
                message,
                n_results=3,
                sections=guess_sections,
                extraction_info=extraction_info,
            )
            if context and not context.startswith("No relevant information found"):
                return context
        except Exception as e:
            logger.warning("[Router] RAG 일반 검색 오류: %s", e)

        if guess_sections:
            try:
                context_fallback = self.rag_pipeline.get_context(
                    message, n_results=3, sections=None, extraction_info=extraction_info
                )
                if context_fallback and not context_fallback.startswith(
                    "No relevant information found"
                ):
                    return context_fallback
            except Exception as fallback_error:
                logger.error("[Router] RAG 재시도 실패: %s", fallback_error)

        logger.info("[Router] RAG 결과 없음 — 빈 컨텍스트로 Planner 진행")
        return None

    def _build_evidence(
        self, search_query: str, rag_context: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not rag_context:
            return []
        evidence: List[Dict[str, Any]] = []
        try:
            hits = self.rag_pipeline.search(search_query, n_results=3)
            for hit in hits or []:
                metadata = hit.get("metadata") or {}
                document = hit.get("document") or ""
                source_id = (
                    metadata.get("vin")
                    or metadata.get("car_id")
                    or metadata.get("id")
                    or ""
                )
                snippet = document[:500] if len(document) > 500 else document
                if snippet:
                    evidence.append(
                        {"source_id": str(source_id), "snippet_text": snippet}
                    )
        except Exception as e:
            logger.warning("[Router] evidence 검색 실패, rag_context 사용: %s", e)

        if not evidence:
            vin_matches = re.findall(r"\b([A-Z0-9]{17})\b", rag_context, re.IGNORECASE)
            source_id = vin_matches[0].upper() if vin_matches else ""
            evidence.append(
                {
                    "source_id": source_id,
                    "snippet_text": rag_context[:500],
                }
            )
        return evidence
