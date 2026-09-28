"""
Session-level conversation logging utilities.

피드백에서 요구한 세션 로그 JSON 스키마를 만족시키기 위한 모델/스토어:

- session_id
- conversation: [{turn_id, speaker, text, timestamp}]  (최종 UI 기준)
- evidence: assistant_turn_id -> [{source_id, snippet_text}]
- latency_ms: assistant_turn_id -> latency_ms
- token_usage: assistant_turn_id -> {input, output, total}
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ConversationTurn:
    turn_id: str
    speaker: str  # "user" | "assistant"
    text: str
    timestamp: str  # ISO8601 string
    lead_status: Optional[str] = (
        None  # Lead status for assistant turns (e.g., "Cold Lead", "Warm Lead", "Hot Lead")
    )


@dataclass
class EvidenceItem:
    source_id: str
    snippet_text: str


@dataclass
class SessionLog:
    session_id: str
    conversation: List[ConversationTurn] = field(default_factory=list)
    evidence: Dict[str, List[EvidenceItem]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    planning_info: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # assistant_turn_id -> planning_info 매핑
    participant_id: Optional[str] = None  # 설문 매칭/중복 사용자 처리용
    user_goal_segment: Optional[str] = None  # 사용자 목표 세그먼트


class SessionLogStore:
    """
    In-memory 세션 로그 저장소.

    세션 로그 저장 전략:
    - 메모리: 활성 세션의 대화 로그 임시 저장
    - 파일: 세션 종료 시 JSON 파일로 영구 저장
    - Redis TTL: SessionExpiryListener가 만료 이벤트 감지 → 자동 저장

    실제 운영에서 외부 스토리지(DB, 로그 파이프라인 등)로 대체할 수 있도록,
    export 시에는 순수 dict 구조만 반환한다.

    세션 만료 처리:
    1. Redis TTL 만료 → SessionExpiryListener가 save_to_file() 호출
    2. BaseAgent가 5분마다 cleanup_and_save_expired_sessions() 호출 (백업)
    3. 명시적 세션 종료 (/session-log/{session_id}/save) → force_save_and_delete()
    """

    _store: Dict[str, SessionLog] = {}
    _assistant_turn_counters: Dict[str, int] = {}
    _session_last_access: Dict[str, float] = {}  # 세션별 마지막 접근 시간

    # 세션 TTL (초 단위) - 30분 (RedisSessionManager, BaseAgent와 동일)
    SESSION_TTL = int(os.getenv("SESSION_TTL", 60 * 30))  # 30분

    # 로그 파일 저장 디렉토리
    # 환경 변수 우선순위: SESSION_LOG_DIR > LOG_DIR/session_logs > 기본값 (backend/logs/session_logs)
    # Docker 환경 감지: /backend 경로가 존재하는지 확인
    _is_docker = os.path.exists("/backend")

    _log_dir_env = os.getenv("SESSION_LOG_DIR")
    if _log_dir_env:
        # Docker 경로(/backend/)가 환경 변수에 있지만 실제로는 로컬 환경인 경우
        if not _is_docker and _log_dir_env.startswith("/backend"):
            # 로컬 환경에서는 프로젝트 루트 기준으로 변환
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parent.parent  # utils -> backend
            LOG_DIR = backend_dir / "logs" / "session_logs"
        else:
            LOG_DIR = Path(_log_dir_env)
    else:
        # LOG_DIR 환경 변수가 있으면 그 하위에 session_logs 생성
        base_log_dir = os.getenv("LOG_DIR")
        if base_log_dir:
            # Docker 경로(/backend/)가 환경 변수에 있지만 실제로는 로컬 환경인 경우
            if not _is_docker and base_log_dir.startswith("/backend"):
                # 로컬 환경에서는 프로젝트 루트 기준으로 변환
                current_file = Path(__file__).resolve()
                backend_dir = current_file.parent.parent  # utils -> backend
                LOG_DIR = backend_dir / "logs" / "session_logs"
            else:
                LOG_DIR = Path(base_log_dir) / "session_logs"
        else:
            # 로컬 개발 환경: backend 디렉토리 기준으로 logs/session_logs 생성
            # 현재 파일 위치: web-client/backend/utils/session_log.py
            # backend 디렉토리: web-client/backend/
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parent.parent  # utils -> backend
            LOG_DIR = backend_dir / "logs" / "session_logs"

    # -------- 세션/턴 관리 --------
    @classmethod
    def get_or_create(cls, session_id: str) -> SessionLog:
        if session_id not in cls._store:
            cls._store[session_id] = SessionLog(session_id=session_id)
            cls._assistant_turn_counters[session_id] = 0
        # 마지막 접근 시간 업데이트
        cls._session_last_access[session_id] = time.time()
        return cls._store[session_id]

    @classmethod
    def get_next_assistant_turn_id(cls, session_id: str) -> str:
        """
        assistant 답변 턴에 사용할 고유한 turn_id 생성.
        숫자 카운터 기반이지만, 문자열로 노출하여 외부 스키마와 유연하게 매핑 가능하게 함.

        이 메서드는 세션을 자동으로 생성합니다 (get_or_create 호출).
        """
        log = cls.get_or_create(session_id)  # 세션 자동 생성
        current = cls._assistant_turn_counters.get(session_id, 0)
        cls._assistant_turn_counters[session_id] = current + 1
        # "a-0", "a-1" ... 형태로 명시적으로 assistant 턴임을 표시
        return f"a-{current}"

    # -------- conversation --------
    @classmethod
    def add_turn(cls, session_id: str, turn: ConversationTurn) -> None:
        log = cls.get_or_create(session_id)
        log.conversation.append(turn)
        # 마지막 접근 시간 업데이트 (세션이 활성 상태임을 표시)
        cls._session_last_access[session_id] = time.time()

    # -------- evidence --------
    @classmethod
    def set_evidence(
        cls,
        session_id: str,
        assistant_turn_id: str,
        items: List[EvidenceItem],
    ) -> None:
        log = cls.get_or_create(session_id)
        log.evidence[assistant_turn_id] = items
        # 마지막 접근 시간 업데이트
        cls._session_last_access[session_id] = time.time()

    # -------- latency --------
    @classmethod
    def set_latency(
        cls,
        session_id: str,
        assistant_turn_id: str,
        latency_ms: float,
    ) -> None:
        log = cls.get_or_create(session_id)
        log.latency_ms[assistant_turn_id] = float(latency_ms or 0.0)
        # 마지막 접근 시간 업데이트
        cls._session_last_access[session_id] = time.time()

    # -------- token usage --------
    @classmethod
    def set_token_usage(
        cls,
        session_id: str,
        assistant_turn_id: str,
        usage: Dict[str, int],
    ) -> None:
        log = cls.get_or_create(session_id)
        # 최소한의 키만 강제
        safe_usage = {
            "input": int(usage.get("input", 0)),
            "output": int(usage.get("output", 0)),
            "total": int(usage.get("total", 0)),
        }
        log.token_usage[assistant_turn_id] = safe_usage
        # 마지막 접근 시간 업데이트
        cls._session_last_access[session_id] = time.time()

    @classmethod
    def set_planning_info(
        cls,
        session_id: str,
        assistant_turn_id: str,
        all_plans: List[Dict[str, Any]],
        selected_plan_id: str,
        selection_reasoning: Optional[str] = None,
    ) -> None:
        """
        planning_info 설정 (assistant_turn_id -> planning_info 매핑)

        Args:
            session_id: 세션 ID
            assistant_turn_id: assistant turn ID (예: "a-0")
            all_plans: 모든 생성된 plans 리스트 (최대 3개)
            selected_plan_id: 선택된 plan ID (예: "A", "B", "C")
            selection_reasoning: 선택 이유 (선택적)
        """
        log = cls.get_or_create(session_id)

        # plan 정보에서 필요한 필드만 추출 (spin_stage, nudge_technique 포함)
        plans_summary = []
        for plan in all_plans:
            plan_summary = {
                "plan_id": plan.get("plan_id", ""),
                "spin_stage": plan.get("spin_stage", ""),
                "nudge_technique": plan.get("nudge_technique", ""),
                "strategic_approach": plan.get("strategic_approach", ""),
                "why_this_strategy": plan.get("why_this_strategy", ""),
            }
            plans_summary.append(plan_summary)

        planning_info = {
            "all_plans": plans_summary,
            "selected_plan_id": selected_plan_id,
        }

        if selection_reasoning:
            planning_info["selection_reasoning"] = selection_reasoning

        log.planning_info[assistant_turn_id] = planning_info
        # 마지막 접근 시간 업데이트
        cls._session_last_access[session_id] = time.time()

    # -------- 메타데이터 필드 --------
    @classmethod
    def set_participant_id(cls, session_id: str, participant_id: Optional[str]) -> None:
        """participant_id 설정 (설문 매칭/중복 사용자 처리용)"""
        log = cls.get_or_create(session_id)
        log.participant_id = participant_id

    @classmethod
    def set_user_goal_segment(
        cls, session_id: str, user_goal_segment: Optional[str]
    ) -> None:
        """user_goal_segment 설정 (사용자 목표 세그먼트)"""
        log = cls.get_or_create(session_id)
        log.user_goal_segment = user_goal_segment

    # -------- 파일 저장 --------
    @classmethod
    def save_to_file(
        cls, session_id: str, log_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        세션 로그를 JSON 파일로 저장.
        같은 session_id는 하나의 파일에 저장되며, 기존 파일이 있으면 데이터를 병합합니다.

        Args:
            session_id: 세션 ID
            log_data: 저장할 로그 데이터 (None이면 현재 메모리에서 가져옴)

        Returns:
            저장된 파일 경로 (실패 시 None)
        """
        if log_data is None:
            log_data = cls.to_dict(session_id)
            if not log_data:
                return None

        # 로그 디렉토리 생성 (상위 디렉토리도 함께 생성)
        try:
            cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # 디렉토리 생성 실패 시 에러 로깅 (상위 레이어에서 처리)
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to create session log directory {cls.LOG_DIR}: {e}. "
                f"Please check permissions or set SESSION_LOG_DIR environment variable."
            )
            return None

        # participant_id와 session_id가 모두 같은 경우에만 기존 세션 로그 파일을 찾아서 병합
        #
        # A/B 테스트 참고:
        # - page.tsx는 "chat_participant_id" 키 사용
        # - vanilla/page.tsx는 "vanilla_chat_participant_id" 키 사용
        # - session_id는 쿠키 기반이므로 페이지가 다르면 다른 session_id가 생성됨
        # - 따라서 같은 participant_id(이메일)를 사용하더라도 session_id가 다르면
        #   자동으로 다른 파일(session_log_{session_id}.json)로 저장되어 A/B 테스트 구분됨
        participant_id = log_data.get("participant_id")
        existing_filepath = None
        existing_json_data = None

        # participant_id와 session_id가 모두 같은 경우에만 기존 파일 찾기
        if participant_id:
            import logging

            logger = logging.getLogger(__name__)
            try:
                for file in cls.LOG_DIR.glob("session_log_*.json"):
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            existing_json = json.load(f)
                            # participant_id와 session_id가 모두 같아야 병합
                            if (
                                existing_json.get("participant_id") == participant_id
                                and existing_json.get("session_id") == session_id
                            ):
                                existing_filepath = file
                                existing_json_data = existing_json
                                logger.debug(
                                    f"Found existing session log for participant_id={participant_id} "
                                    f"and session_id={session_id}: {file.name}"
                                )
                                break
                    except Exception:
                        # 파일 읽기 실패 시 무시하고 다음 파일 확인
                        continue
            except Exception as e:
                logger.warning(f"Error while searching for existing session logs: {e}")

        # 파일명: participant_id와 session_id가 모두 같은 기존 파일이 있으면 그 파일 사용, 없으면 session_id 기반
        if existing_filepath and existing_json_data:
            filepath = existing_filepath
            target_session_id = existing_json_data.get("session_id", session_id)
            # participant_id와 session_id 기반 병합임을 로깅
            logger.info(
                f"📋 Merging session {session_id} into existing file for "
                f"participant_id={participant_id} and session_id={session_id}. "
                f"Target file: {filepath.name} (original session_id: {target_session_id})"
            )
        else:
            filename = f"session_log_{session_id}.json"
            filepath = cls.LOG_DIR / filename
            target_session_id = session_id

        # 기존 파일이 있으면 로드하여 병합
        # 이미 participant_id 기반으로 찾은 파일이 있으면 그 데이터 사용 (중복 로딩 방지)
        existing_data = None
        if existing_filepath and existing_json_data:
            # 이미 로드한 데이터 사용
            existing_data = existing_json_data
        elif filepath.exists():
            # 새 파일이거나 기존 파일을 다시 로드해야 하는 경우
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to load existing session log file {filepath}: {e}. "
                    f"Will overwrite with new data."
                )

        # 기존 데이터와 병합
        if existing_data:
            # conversation 병합: turn_id 기준으로 중복 제거하고 새 데이터 추가
            existing_turn_ids = {
                turn.get("turn_id") for turn in existing_data.get("conversation", [])
            }
            new_conversation = [
                turn
                for turn in log_data.get("conversation", [])
                if turn.get("turn_id") not in existing_turn_ids
            ]
            merged_conversation = (
                existing_data.get("conversation", []) + new_conversation
            )

            # conversation을 타임스탬프 기준으로 정렬 (시간순 정렬)
            # 타임스탬프가 없는 경우 맨 뒤로 이동
            def get_timestamp(turn):
                ts = turn.get("timestamp", "")
                if not ts:
                    return "9999-12-31T23:59:59"  # 타임스탬프 없는 항목은 맨 뒤로
                return ts

            merged_conversation.sort(key=get_timestamp)

            # evidence, latency_ms, token_usage, planning_info 병합: 새 값으로 업데이트
            # session_id는 기존 파일의 session_id를 유지 (첫 세션 ID)
            merged_data = {
                "session_id": existing_data.get("session_id", target_session_id),
                "conversation": merged_conversation,
                "evidence": {
                    **existing_data.get("evidence", {}),
                    **log_data.get("evidence", {}),
                },
                "latency_ms": {
                    **existing_data.get("latency_ms", {}),
                    **log_data.get("latency_ms", {}),
                },
                "token_usage": {
                    **existing_data.get("token_usage", {}),
                    **log_data.get("token_usage", {}),
                },
                "planning_info": {
                    **existing_data.get("planning_info", {}),
                    **log_data.get("planning_info", {}),
                },
                "participant_id": log_data.get("participant_id")
                or existing_data.get("participant_id"),
                "user_goal_segment": log_data.get("user_goal_segment")
                or existing_data.get("user_goal_segment"),
            }
        else:
            # 새 데이터도 타임스탬프 기준으로 정렬
            conversation = log_data.get("conversation", [])
            if conversation:

                def get_timestamp(turn):
                    ts = turn.get("timestamp", "")
                    if not ts:
                        return "9999-12-31T23:59:59"
                    return ts

                conversation.sort(key=get_timestamp)
                log_data["conversation"] = conversation
            merged_data = log_data

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            import logging

            logger = logging.getLogger(__name__)
            # 파일명과 실제 저장된 session_id가 다를 수 있음 (participant_id 병합 시)
            if target_session_id != session_id:
                logger.info(
                    f"Successfully saved session log: session_id={session_id} -> "
                    f"file={filepath.name} (merged with session_id={target_session_id}, "
                    f"participant_id={participant_id})"
                )
            else:
                logger.info(f"Successfully saved session log to {filepath}")

            # participant_id 기반 병합인 경우, 다른 session_id로 저장된 파일이 있으면 삭제
            if (
                existing_filepath
                and existing_filepath.name != f"session_log_{session_id}.json"
            ):
                other_filepath = cls.LOG_DIR / f"session_log_{session_id}.json"
                if other_filepath.exists() and other_filepath != existing_filepath:
                    try:
                        other_filepath.unlink()
                        logger.info(
                            f"Removed duplicate session log file: {other_filepath.name}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove duplicate session log file {other_filepath}: {e}"
                        )

            return str(filepath)
        except Exception as e:
            # 상세한 에러 로깅
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to save session log to file {filepath}: {e}. "
                f"LOG_DIR: {cls.LOG_DIR}, exists: {cls.LOG_DIR.exists()}, "
                f"is_dir: {cls.LOG_DIR.is_dir() if cls.LOG_DIR.exists() else 'N/A'}"
            )
            return None

    @classmethod
    def cleanup_and_save_expired_sessions(cls) -> List[str]:
        """
        만료된 세션을 파일로 저장하고 메모리에서 삭제.

        Returns:
            저장된 세션 ID 리스트
        """
        import logging

        logger = logging.getLogger(__name__)

        now = time.time()

        # 디버그: 현재 추적 중인 세션들
        logger.debug(
            f"[SessionLogStore] Checking for expired sessions. "
            f"Current sessions in memory: {list(cls._store.keys())}, "
            f"Sessions with last_access: {list(cls._session_last_access.keys())}, "
            f"SESSION_TTL: {cls.SESSION_TTL}s ({cls.SESSION_TTL/60:.1f}min)"
        )

        expired_sessions = [
            sid
            for sid, last_access in cls._session_last_access.items()
            if now - last_access > cls.SESSION_TTL
        ]

        if expired_sessions:
            logger.info(
                f"[SessionLogStore] Found {len(expired_sessions)} expired sessions: {expired_sessions}"
            )
        else:
            logger.debug(f"[SessionLogStore] No expired sessions found")

        saved_sessions = []
        for sid in expired_sessions:
            last_access = cls._session_last_access.get(sid, 0)
            age_minutes = (now - last_access) / 60
            logger.info(
                f"[SessionLogStore] Processing expired session: {sid} "
                f"(age: {age_minutes:.1f} minutes)"
            )

            log_data = cls.to_dict(sid)
            if log_data:
                filepath = cls.save_to_file(sid, log_data)
                if filepath:
                    saved_sessions.append(sid)
                    logger.info(
                        f"[SessionLogStore] Successfully saved session {sid} to {filepath}"
                    )
                else:
                    logger.error(f"[SessionLogStore] Failed to save session {sid}")
            else:
                logger.warning(
                    f"[SessionLogStore] No data found for expired session {sid}"
                )

            # 메모리에서 삭제
            cls.delete(sid)

        return saved_sessions

    @classmethod
    def force_save_and_delete(cls, session_id: str) -> Optional[str]:
        """
        세션 로그를 강제로 파일로 저장하고 메모리에서 삭제.
        (명시적 세션 종료 시 사용)

        Returns:
            저장된 파일 경로 (실패 시 None)
        """
        log_data = cls.to_dict(session_id)
        if not log_data:
            # 세션이 메모리에 없음
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Session {session_id} not found in memory. "
                f"Available sessions: {list(cls._store.keys())[:10]}"
            )
            return None

        filepath = cls.save_to_file(session_id, log_data)
        if filepath:
            # 파일 저장 성공 시에만 메모리에서 삭제
            cls.delete(session_id)
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Session {session_id} saved and deleted from memory")
        else:
            # 파일 저장 실패 시 에러 로깅
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to save session {session_id} to file. "
                f"Session will remain in memory. LOG_DIR: {cls.LOG_DIR}"
            )
        return filepath

    # -------- export / 관리 --------
    @classmethod
    def to_dict(cls, session_id: str) -> Dict[str, Any]:
        log = cls._store.get(session_id)
        if not log:
            return {}
        # dataclass → dict 변환
        return asdict(log)

    @classmethod
    def list_all_sessions(cls) -> List[str]:
        """현재 메모리에 있는 모든 세션 ID 리스트 반환"""
        return list(cls._store.keys())

    @classmethod
    def batch_export(
        cls, session_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 세션을 한 번에 export (메모리에서 읽기만, 파일 저장 없음).

        Args:
            session_ids: export할 세션 ID 리스트 (None이면 모든 세션)

        Returns:
            {session_id: log_data, ...} 형태의 딕셔너리
        """
        if session_ids is None:
            session_ids = cls.list_all_sessions()

        result = {}
        for sid in session_ids:
            log_data = cls.to_dict(sid)
            if log_data:
                result[sid] = log_data

        return result

    @classmethod
    def batch_save_to_file(
        cls,
        session_ids: Optional[List[str]] = None,
        delete_after_save: bool = False,
    ) -> Dict[str, Any]:
        """
        여러 세션을 한 번에 파일로 저장.

        Args:
            session_ids: 저장할 세션 ID 리스트 (None이면 모든 세션)
            delete_after_save: 저장 후 메모리에서 삭제할지 여부

        Returns:
            {
                "saved_count": 3,
                "failed_count": 1,
                "saved_sessions": [
                    {"session_id": "session_1", "filepath": "..."},
                    ...
                ],
                "failed_sessions": ["session_2", ...]
            }
        """
        if session_ids is None:
            session_ids = cls.list_all_sessions()

        saved_sessions = []
        failed_sessions = []

        for sid in session_ids:
            log_data = cls.to_dict(sid)
            if not log_data:
                failed_sessions.append(sid)
                continue

            filepath = cls.save_to_file(sid, log_data)
            if filepath:
                saved_sessions.append({"session_id": sid, "filepath": filepath})
                if delete_after_save:
                    cls.delete(sid)
            else:
                failed_sessions.append(sid)

        return {
            "saved_count": len(saved_sessions),
            "failed_count": len(failed_sessions),
            "saved_sessions": saved_sessions,
            "failed_sessions": failed_sessions,
        }

    @classmethod
    def delete(cls, session_id: str) -> None:
        cls._store.pop(session_id, None)
        cls._assistant_turn_counters.pop(session_id, None)
        cls._session_last_access.pop(session_id, None)
