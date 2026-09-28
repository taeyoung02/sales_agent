# 모든 Agent가 공유하는 공통 기능을 제공하는 Base 클래스

import os
import re
import time
import logging
from typing import List, Dict, Optional, Any
from rag.rag import get_rag_pipeline
from prompts import (
    get_sales_knowledge,
    get_common_tool_guidelines,
    TOOL_REGISTRY,
)
from services.llm import LLMClientFactory
from utils.session_log import SessionLogStore

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    모든 Agent가 공유하는 공통 기능을 제공하는 Base 클래스

    세션 관리 전략:
    - BaseAgent: 메모리 기반 세션 메타데이터 (user_info, conversation_summaries 등)
    - SessionLogStore: 세션 대화 로그 (JSON 파일로 자동 저장)
    - RedisSessionManager: Redis 기반 세션 TTL 관리 (프로덕션 환경)

    세션 만료 처리:
    - Redis TTL 만료 시 SessionExpiryListener가 자동으로 로그 파일 저장
    - BaseAgent는 5분마다 메모리 정리 (cleanup)하여 orphan 데이터 제거
    """

    # 세션별 사용자 정보 저장소 (전체 pipeline에서 공유)
    _user_info_store: Dict[str, Dict[str, Any]] = {}

    # 세션별 대화 히스토리 요약본 저장소
    _conversation_summaries: Dict[str, str] = {}

    # 세션별 요약본 버전 (업데이트 추적용)
    _conversation_summary_versions: Dict[str, int] = {}

    # 세션별 마지막 요약 생성 시점의 메시지 개수
    _conversation_summary_message_counts: Dict[str, int] = {}

    # 세션별 마지막 접근 시간 (TTL 정리용)
    _session_last_access: Dict[str, float] = {}

    # 기본 세션 ID (session_id가 없을 때 사용)
    _default_session_id = "default"

    # 요약 업데이트 임계값 (메시지 개수)
    SUMMARY_UPDATE_THRESHOLD = 15  # 15개 메시지마다 요약 업데이트
    RECENT_MESSAGES_COUNT = 8  # 최근 8개 메시지는 그대로 사용

    # 세션 TTL (초 단위) - 30분 (RedisSessionManager와 동일)
    # 환경 변수는 문자열이므로 int로 변환
    SESSION_TTL = int(os.getenv("SESSION_TTL", 60 * 30))  # 30분

    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = LLMClientFactory.create_client()
        self.rag_pipeline = get_rag_pipeline()

    # 마지막 세션 정리 시각 추적 (클래스 변수)
    _last_cleanup_time: float = 0

    # 세션 정리 주기 (초 단위) - 5분마다 정리
    CLEANUP_INTERVAL = 60 * 5

    @classmethod
    def _touch_session(cls, session_id: Optional[str] = None) -> None:
        """
        세션 접근 시간 업데이트 및 만료된 세션 정리

        Args:
            session_id: 세션 ID
        """
        sid = session_id or cls._default_session_id
        now = time.time()

        # 세션 접근 시간 업데이트
        cls._session_last_access[sid] = now

        # 주기적으로 만료된 세션 정리 (마지막 정리 후 CLEANUP_INTERVAL 이상 경과 시)
        if now - cls._last_cleanup_time >= cls.CLEANUP_INTERVAL:
            cls._last_cleanup_time = now
            cls._cleanup_expired_sessions()

    @classmethod
    def _cleanup_expired_sessions(cls) -> None:
        """
        만료된 세션 데이터 정리 (TTL 초과)
        """
        now = time.time()
        expired_sessions = []

        for sid, last_access in cls._session_last_access.items():
            # 타입 체크: last_access가 숫자가 아니면 제거
            if not isinstance(last_access, (int, float)):
                logger.warning(
                    f"[BaseAgent] Invalid last_access type for session {sid}: {type(last_access)}, "
                    f"value: {last_access}. Removing invalid entry."
                )
                expired_sessions.append(sid)
                continue

            # 만료된 세션 확인
            if now - last_access > cls.SESSION_TTL:
                expired_sessions.append(sid)

        if expired_sessions:
            logger.info(
                f"[BaseAgent] Starting cleanup for {len(expired_sessions)} expired sessions: {expired_sessions}"
            )

        for sid in expired_sessions:
            cls._user_info_store.pop(sid, None)
            cls._conversation_summaries.pop(sid, None)
            cls._conversation_summary_versions.pop(sid, None)
            cls._conversation_summary_message_counts.pop(sid, None)
            cls._session_last_access.pop(sid, None)
            logger.info(f"[BaseAgent] Expired session cleaned: {sid}")

        # 세션 메타데이터 정리와 함께, 세션 로그도 TTL 기준으로 파일로 저장 후 정리
        try:
            logger.debug(
                f"[BaseAgent] Calling SessionLogStore.cleanup_and_save_expired_sessions()..."
            )
            saved_sessions = SessionLogStore.cleanup_and_save_expired_sessions()
            if saved_sessions:
                logger.info(
                    f"[BaseAgent] Session logs saved and cleaned for expired sessions: {saved_sessions}"
                )
            else:
                logger.debug(f"[BaseAgent] No expired session logs to save")
        except Exception as e:
            # 로그 저장 실패는 치명적 오류가 아니므로 경고만 남김
            logger.warning(
                f"[BaseAgent] Failed to cleanup/save expired session logs: {e}",
                exc_info=True,
            )

    @classmethod
    def get_user_info(cls, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        세션별 사용자 정보 조회

        Args:
            session_id: 세션 ID (None이면 기본 세션 사용)

        Returns:
            사용자 정보 Dict
        """
        cls._touch_session(session_id)  # 세션 접근 시간 업데이트

        sid = session_id or cls._default_session_id
        if sid not in cls._user_info_store:
            cls._user_info_store[sid] = {
                "purchase_purpose": [],
                "preferred_car_type": None,
                "budget_krw": None,
                "lead_status": None,
            }
        return cls._user_info_store[sid]

    @classmethod
    def update_user_info(
        cls, user_info: Dict[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        세션별 사용자 정보 업데이트 (누적)

        Args:
            user_info: 업데이트할 사용자 정보
            session_id: 세션 ID (None이면 기본 세션 사용)

        Returns:
            업데이트된 사용자 정보 Dict
        """
        sid = session_id or cls._default_session_id
        current_info = cls.get_user_info(sid)

        # purchase_purpose 누적 (중복 제거)
        if isinstance(user_info.get("purchase_purpose"), list):
            existing = set(current_info["purchase_purpose"])
            new_purposes = [
                p for p in user_info["purchase_purpose"] if p not in existing
            ]
            current_info["purchase_purpose"].extend(new_purposes)

        # preferred_car_type 업데이트 (명시된 경우만)
        if user_info.get("preferred_car_type"):
            current_info["preferred_car_type"] = user_info["preferred_car_type"]

        # budget_krw 업데이트 (명시된 경우만)
        if user_info.get("budget_krw"):
            try:
                budget = float(user_info["budget_krw"])
                if budget > 0:
                    current_info["budget_krw"] = budget
            except (ValueError, TypeError):
                pass

        # lead_status 업데이트 (명시된 경우만)
        if user_info.get("lead_status"):
            current_info["lead_status"] = user_info["lead_status"]

        cls._user_info_store[sid] = current_info

        return current_info

    @classmethod
    def clear_user_info(cls, session_id: Optional[str] = None) -> None:
        """
        세션별 사용자 정보 초기화

        Args:
            session_id: 세션 ID (None이면 기본 세션 사용)
        """
        sid = session_id or cls._default_session_id
        if sid in cls._user_info_store:
            cls._user_info_store[sid] = {
                "purchase_purpose": [],
                "preferred_car_type": None,
                "budget_krw": None,
                "lead_status": None,
            }

    def get_planner_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Planner Agent 전용 Tool 정의 반환
        Planner는 3D 뷰어 제어 도구 + generatePresentationScript만 계획한다.
        차량 사실은 Router RAG를 사용한다.
        """
        return [
            TOOL_REGISTRY["loadVehicle"]["definition"],
            TOOL_REGISTRY["generateHeatmap"]["definition"],
            TOOL_REGISTRY["setCamera"]["definition"],
            TOOL_REGISTRY["generatePresentationScript"]["definition"],
        ]

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Executor Agent용 Tool 정의 반환
        모든 tool 정의
        """
        return [TOOL_REGISTRY[tool_name]["definition"] for tool_name in TOOL_REGISTRY]

    def get_common_sales_knowledge(self) -> str:
        """
        모든 Agent가 공유하는 세일즈 지식 프롬프트 반환
        프롬프트 모듈에서 가져옴
        """
        return get_sales_knowledge()

    def get_common_tool_guidelines(self) -> str:
        """
        모든 Agent가 공유하는 공통 Tool 사용 가이드라인 반환
        프롬프트 모듈에서 가져옴
        """
        return get_common_tool_guidelines()

    def extract_vehicle_id_from_history(
        self, conversation_history: Optional[List[Dict[str, str]]]
    ) -> Optional[str]:
        """
        대화 히스토리에서 가장 최근에 로드된 vehicle_id를 추출

        Assistant 응답에서 "(현재 로드된 차량 ID: ...)" 패턴을 찾습니다.
        """
        if not conversation_history:
            return None

        # 역순으로 검색하여 가장 최근의 vehicle_id를 찾음
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # 패턴: "(현재 로드된 차량 ID: test)" 또는 "(현재 로드된 차량: test)"
                match = re.search(r"\(현재 로드된 차량(?:\s+ID)?:\s*([^)]+)\)", content)
                if match:
                    return match.group(1).strip()

        return None

    def build_context_section(
        self,
        vehicle_id: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        System Prompt에 추가할 [현재 컨텍스트] 섹션 생성
        """
        # vehicle_id 우선순위: 명시적 전달 > 대화 히스토리에서 추출
        current_vehicle_id = vehicle_id

        if not current_vehicle_id and conversation_history:
            current_vehicle_id = self.extract_vehicle_id_from_history(
                conversation_history
            )

        if not current_vehicle_id:
            return ""

        return f"""
            [현재 컨텍스트]
            - 현재 화면에 로드된 차량 ID: {current_vehicle_id}
            - 사용자가 "이 차량", "해당 차량", "현재 차량", "지금 보는 차량" 등으로 언급할 때는 반드시 이 vehicle_id ({current_vehicle_id})를 사용하세요.
            - generatePresentationScript를 호출할 때 vehicle_id가 필요하면 반드시 '{current_vehicle_id}'를 사용하세요.
            """

    @classmethod
    def _generate_conversation_summary(
        cls,
        conversation_history: List[Dict[str, str]],
        session_id: Optional[str] = None,
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        대화 히스토리를 요약하여 intent를 유지

        Args:
            conversation_history: 전체 대화 히스토리
            session_id: 세션 ID
            existing_summary: 기존 요약본 (있으면 증분 업데이트)

        Returns:
            요약본 문자열
        """
        if not conversation_history or len(conversation_history) == 0:
            return ""

        sid = session_id or cls._default_session_id

        # 요약 프롬프트 구성
        if existing_summary:
            # 증분 업데이트: 기존 요약본 + 새로운 메시지들
            new_messages = conversation_history[
                cls._conversation_summary_message_counts.get(sid, 0) :
            ]
            if not new_messages:
                return existing_summary

            summary_prompt = f"""
다음은 이전 대화 요약본과 새로운 대화 메시지들입니다. 
기존 요약본을 새로운 메시지들을 반영하여 업데이트하세요.

[이전 대화 요약본]
{existing_summary}

[새로운 대화 메시지들]
{chr(10).join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in new_messages])}

위 정보를 바탕으로 업데이트된 대화 요약본을 생성하세요.
요약본에는 다음 정보가 포함되어야 합니다:
- 사용자의 주요 관심사 및 의도
- 수집된 사용자 정보 (구매 목적, 선호 차종, 예산 등)
- 현재 진행 중인 대화의 맥락
- 중요한 결정 사항이나 선호사항

요약본은 간결하되 중요한 정보는 누락하지 마세요.
"""
        else:
            # 전체 요약: 처음부터 요약 생성
            summary_prompt = f"""
다음 대화 히스토리를 요약하여 사용자의 intent와 맥락을 유지할 수 있도록 하세요.

[대화 히스토리]
{chr(10).join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in conversation_history])}

위 대화를 요약하되, 다음 정보를 반드시 포함하세요:
- 사용자의 주요 관심사 및 의도
- 수집된 사용자 정보 (구매 목적, 선호 차종, 예산 등)
- 현재 진행 중인 대화의 맥락
- 중요한 결정 사항이나 선호사항
- 언급된 차량 정보나 비교 대상

요약본은 간결하되 중요한 정보는 누락하지 마세요.
"""

        try:
            # LLM으로 요약 생성 (빠른 모델 사용)
            from services.llm import LLMClientFactory

            llm_client = LLMClientFactory.create_client()

            messages = [
                {
                    "role": "user",
                    "content": summary_prompt,
                }
            ]

            summary = llm_client.generate_text(
                messages=messages,
                system_instruction="너는 대화 히스토리를 요약하여 intent를 유지하는 전문가다.",
                use_fast_model=True,  # 요약은 빠른 모델 사용
            )

            return summary.strip()
        except Exception as e:
            logger.error(f"요약 생성 오류: {e}", exc_info=True)
            # 오류 시 기존 요약본 반환 또는 빈 문자열
            return existing_summary or ""

    @classmethod
    def _update_summary_if_needed(
        cls,
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None,
    ) -> None:
        """
        필요시 대화 히스토리 요약본 업데이트

        Args:
            conversation_history: 전체 대화 히스토리
            session_id: 세션 ID
        """
        if not conversation_history:
            return

        sid = session_id or cls._default_session_id
        current_count = len(conversation_history)
        last_summary_count = cls._conversation_summary_message_counts.get(sid, 0)

        # 요약 업데이트 조건: 메시지가 임계값 이상 증가했을 때
        if current_count - last_summary_count >= cls.SUMMARY_UPDATE_THRESHOLD:
            existing_summary = cls._conversation_summaries.get(sid, "")
            new_summary = cls._generate_conversation_summary(
                conversation_history=conversation_history,
                session_id=sid,
                existing_summary=existing_summary if existing_summary else None,
            )

            if new_summary:
                cls._conversation_summaries[sid] = new_summary
                cls._conversation_summary_versions[sid] = (
                    cls._conversation_summary_versions.get(sid, 0) + 1
                )
                cls._conversation_summary_message_counts[sid] = current_count
                logger.info(
                    f"[BaseAgent] 대화 요약 업데이트 완료 (세션: {sid}, 버전: {cls._conversation_summary_versions[sid]}, 메시지 수: {current_count})"
                )

    @classmethod
    def get_effective_history(
        cls,
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None,
        recent_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        요약본 + 최근 N개 메시지 조합하여 효과적인 히스토리 반환

        Args:
            conversation_history: 전체 대화 히스토리
            session_id: 세션 ID
            recent_count: 최근 메시지 개수 (기본값: RECENT_MESSAGES_COUNT)

        Returns:
            요약본 메시지 + 최근 N개 메시지 조합
        """
        if not conversation_history:
            return []

        cls._touch_session(session_id)  # 세션 접근 시간 업데이트

        sid = session_id or cls._default_session_id
        recent_count = recent_count or cls.RECENT_MESSAGES_COUNT

        # 요약본 업데이트 확인
        cls._update_summary_if_needed(conversation_history, sid)

        # 요약본 가져오기
        summary = cls._conversation_summaries.get(sid, "")

        # 최근 메시지들
        recent_messages = (
            conversation_history[-recent_count:]
            if len(conversation_history) > recent_count
            else conversation_history
        )

        # 요약본이 있으면 요약본을 첫 번째 메시지로 추가
        effective_history = []
        if summary:
            effective_history.append(
                {
                    "role": "system",
                    "content": f"[이전 대화 요약]\n{summary}\n\n위 요약본은 이전 대화의 맥락과 intent를 유지하기 위한 것입니다. 최근 메시지들과 함께 참고하세요.",
                }
            )

        # 최근 메시지들 추가
        effective_history.extend(recent_messages)

        return effective_history

    @classmethod
    def clear_conversation_summary(cls, session_id: Optional[str] = None) -> None:
        """
        세션별 대화 요약본 초기화

        Args:
            session_id: 세션 ID (None이면 기본 세션 사용)
        """
        sid = session_id or cls._default_session_id
        if sid in cls._conversation_summaries:
            del cls._conversation_summaries[sid]
        if sid in cls._conversation_summary_versions:
            del cls._conversation_summary_versions[sid]
        if sid in cls._conversation_summary_message_counts:
            del cls._conversation_summary_message_counts[sid]
