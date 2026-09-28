"""
Context variable management for session ID in logging
"""

from contextvars import ContextVar
import logging

# Context variable for session ID
session_id_context: ContextVar[str] = ContextVar("session_id", default="unknown")


class SessionIdFilter(logging.Filter):
    """세션 ID를 로그 레코드에 추가하는 필터"""

    def filter(self, record):
        """로그 레코드에 session_id 속성 추가"""
        record.session_id = session_id_context.get()
        return True


def set_session_id(session_id: str) -> None:
    """세션 ID를 컨텍스트에 설정"""
    session_id_context.set(session_id)


def get_session_id() -> str:
    """현재 컨텍스트의 세션 ID 가져오기"""
    return session_id_context.get()
