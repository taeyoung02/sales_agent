"""
OpenAI LLM Client Implementation with usage & latency logging
"""

import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
from openai import OpenAI
import logging

from .llm_client_interface import LLMClient
from .llm_usage_tracker import LLMUsageTracker, LLMCallTimer


logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """OpenAI LLM 클라이언트 구현"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
    ):
        """
        Args:
            api_key: OpenAI API 키
            default_model: 기본 모델 이름
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model

        # 빠른 모델 설정
        self.fast_model = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")

    def get_provider_name(self) -> str:
        return "openai"

    def get_default_model(self) -> str:
        return os.getenv("OPENAI_CHAT_MODEL", self.default_model)

    def _resolve_model(self, model: Optional[str], use_fast_model: bool) -> str:
        """
        모델 이름 결정: use_fast_model이 True면 빠른 모델, 아니면 model 또는 기본값
        """
        if use_fast_model:
            return self.fast_model
        return model or self.get_default_model()

    def _log_usage_and_latency(
        self,
        *,
        session_id: Optional[str],
        model_name: str,
        response: Any,
        latency_ms: float,
        call_type: str,
    ) -> None:
        """
        OpenAI 응답 객체에서 usage 정보를 추출하여 LLMUsageTracker에 기록하고,
        상세 디버그 로그를 남긴다.
        """
        usage = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if usage is not None:
            # OpenAI Responses API의 usage 필드가 dict 형식 또는 객체 형식일 수 있으므로 안전하게 접근
            try:
                input_tokens = (
                    getattr(usage, "input_tokens", 0)
                    or getattr(usage, "prompt_tokens", 0)
                    or 0
                )
                output_tokens = (
                    getattr(usage, "output_tokens", 0)
                    or getattr(usage, "completion_tokens", 0)
                    or 0
                )
                total_tokens = getattr(usage, "total_tokens", 0) or 0

                # 검증: total이 input + output과 일치하는지 확인 (디버깅용)
                if total_tokens > 0 and total_tokens != input_tokens + output_tokens:
                    logger.warning(
                        f"[LLM][OpenAI] Token count mismatch: total={total_tokens}, "
                        f"input={input_tokens}, output={output_tokens}, "
                        f"calculated={input_tokens + output_tokens}"
                    )
            except Exception:
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

        # 세션 단위 토큰 사용량 집계
        if session_id:
            LLMUsageTracker.add_usage(
                session_id=session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # 세부 로그 (세션 ID는 logging_context의 Filter를 통해 자동으로 붙음)
        logger.info(
            "[LLM][OpenAI] call_type=%s model=%s latency_ms=%.2f "
            "input_tokens=%s output_tokens=%s total_tokens=%s session_id=%s",
            call_type,
            model_name,
            latency_ms,
            input_tokens,
            output_tokens,
            total_tokens,
            session_id or "unknown",
        )

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """텍스트 생성"""
        model_name = self._resolve_model(model, use_fast_model)

        # OpenAI는 system 메시지를 messages에 포함
        openai_messages = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(messages)

        with LLMCallTimer() as timer:
            response = self.client.responses.create(
                model=model_name, input=openai_messages, **kwargs
            )

        # usage & latency 로깅
        self._log_usage_and_latency(
            session_id=session_id,
            model_name=model_name,
            response=response,
            latency_ms=timer.elapsed_ms,
            call_type="text",
        )

        return response.output_text

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """Structured Output 생성"""
        model_name = self._resolve_model(model, use_fast_model)

        # OpenAI는 system 메시지를 messages에 포함
        openai_messages = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(messages)

        with LLMCallTimer() as timer:
            response = self.client.responses.parse(
                model=model_name,
                instructions=system_instruction or "",
                input=openai_messages,
                text_format=response_schema,
                **kwargs,
            )

        # usage & latency 로깅
        self._log_usage_and_latency(
            session_id=session_id,
            model_name=model_name,
            response=response,
            latency_ms=timer.elapsed_ms,
            call_type="structured",
        )

        return response.output_parsed
