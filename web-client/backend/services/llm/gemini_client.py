"""
Gemini LLM Client Implementation with usage & latency logging
"""

import os
import json
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
from google import genai
import logging

from .llm_client_interface import LLMClient
from .llm_usage_tracker import LLMUsageTracker, LLMCallTimer


logger = logging.getLogger(__name__)


def clean_json_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gemini API가 인식하지 못하는 필드를 제거한 JSON 스키마 반환
    - additionalProperties 필드 제거
    """
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        elif isinstance(value, dict):
            cleaned[key] = clean_json_schema_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [
                clean_json_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned


class GeminiLLMClient(LLMClient):
    """Gemini LLM 클라이언트 구현"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
    ):
        """
        Args:
            api_key: Gemini API 키
            default_model: 기본 모델 이름
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.default_model = default_model

        # 빠른 모델 설정
        self.fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

    def get_provider_name(self) -> str:
        return "gemini"

    def get_default_model(self) -> str:
        return os.getenv("GEMINI_CHAT_MODEL", self.default_model)

    def _resolve_model(self, model: Optional[str], use_fast_model: bool) -> str:
        """
        모델 이름 결정: use_fast_model이 True면 빠른 모델, 아니면 model 또는 기본값
        """
        if use_fast_model:
            return self.fast_model
        return model or self.get_default_model()

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """메시지를 Gemini API 형식으로 변환"""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            # system은 system_instruction으로 처리

        return contents

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
        Gemini 응답에서 usage 정보를 읽을 수 있으면 기록하고,
        세션 단위 집계 및 디버그 로그를 남긴다.

        현재 google-genai 클라이언트는 OpenAI처럼 일관된 usage 필드를 제공하지 않을 수 있으므로,
        사용 가능할 때만 best-effort로 기록한다.
        """
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        # 향후 library가 usage 메타데이터를 제공할 경우를 대비한 방어적 접근
        usage = getattr(response, "usage_metadata", None) or getattr(
            response, "usage", None
        )
        if usage is not None:
            try:
                # google-genai의 usage_metadata 형태에 맞게 필드명을 조정
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                total_tokens = getattr(usage, "total_token_count", 0) or 0

                # 검증: total이 input + output과 일치하는지 확인 (디버깅용)
                if total_tokens > 0 and total_tokens != input_tokens + output_tokens:
                    logger.warning(
                        f"[LLM][Gemini] Token count mismatch: total={total_tokens}, "
                        f"input={input_tokens}, output={output_tokens}, "
                        f"calculated={input_tokens + output_tokens}"
                    )
            except Exception:
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

        if session_id:
            LLMUsageTracker.add_usage(
                session_id=session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        logger.info(
            "[LLM][Gemini] call_type=%s model=%s latency_ms=%.2f "
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
        contents = self._convert_messages_to_gemini_format(messages)
        model_name = self._resolve_model(model, use_fast_model)

        config: Dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction

        with LLMCallTimer() as timer:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config if config else None,
            )

        self._log_usage_and_latency(
            session_id=session_id,
            model_name=model_name,
            response=response,
            latency_ms=timer.elapsed_ms,
            call_type="text",
        )

        return response.text

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
        contents = self._convert_messages_to_gemini_format(messages)
        model_name = self._resolve_model(model, use_fast_model)

        # JSON 스키마 정리 (additionalProperties 제거)
        cleaned_schema = clean_json_schema_for_gemini(
            response_schema.model_json_schema()
        )

        config: Dict[str, Any] = {
            "response_schema": cleaned_schema,
            "response_mime_type": "application/json",
        }
        if system_instruction:
            config["system_instruction"] = system_instruction

        with LLMCallTimer() as timer:
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

        self._log_usage_and_latency(
            session_id=session_id,
            model_name=model_name,
            response=response,
            latency_ms=timer.elapsed_ms,
            call_type="structured",
        )

        # JSON 파싱 및 Pydantic 모델로 변환
        response_text = response.text
        try:
            response_json = json.loads(response_text)
            return response_schema.model_validate(response_json)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Failed to parse structured output: {e}, Response: {response_text[:200]}"
            )
