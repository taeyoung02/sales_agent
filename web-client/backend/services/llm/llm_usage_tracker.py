"""
LLM Usage & Latency Tracking Utilities

각 LLM 호출에 대한 토큰 사용량과 레이턴시를 세션/턴 기준으로 수집하기 위한 헬퍼.
세션 단위 집계를 담당하고, 실제 세션 로그(JSON export)는 상위 레이어에서
SessionLogStore와 같은 컴포넌트가 소비하도록 설계한다.
"""

from typing import Dict, Optional, Tuple
import time


class LLMUsageTracker:
    """
    세션 단위 토큰 사용량 집계 도우미.

    - add_usage(session_id, input_tokens, output_tokens)
    - get_session_usage(session_id): 누적 사용량 조회 (삭제하지 않음)
    - pop_session_usage(session_id): 누적 사용량 조회 후 초기화
    """

    _usage: Dict[str, Dict[str, int]] = {}

    @classmethod
    def add_usage(
        cls,
        session_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not session_id:
            return

        if session_id not in cls._usage:
            cls._usage[session_id] = {"input": 0, "output": 0, "total": 0}

        usage = cls._usage[session_id]
        usage["input"] += int(input_tokens or 0)
        usage["output"] += int(output_tokens or 0)
        usage["total"] += int(input_tokens or 0) + int(output_tokens or 0)

    @classmethod
    def get_session_usage(cls, session_id: str) -> Dict[str, int]:
        return cls._usage.get(session_id, {"input": 0, "output": 0, "total": 0})

    @classmethod
    def pop_session_usage(cls, session_id: str) -> Dict[str, int]:
        return cls._usage.pop(session_id, {"input": 0, "output": 0, "total": 0})


class LLMCallTimer:
    """
    LLM 호출 레이턴시 측정을 위한 간단한 컨텍스트 매니저.

    with LLMCallTimer() as t:
        ... LLM 호출 ...
    latency_ms = t.elapsed_ms
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "LLMCallTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return (end - self._start) * 1000.0
