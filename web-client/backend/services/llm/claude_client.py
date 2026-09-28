"""
Claude (Anthropic) LLM Client Implementation
"""

import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

from .llm_client_interface import LLMClient


class ClaudeLLMClient(LLMClient):
    """Claude (Anthropic) LLM 클라이언트 구현"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        """
        Args:
            api_key: Anthropic API 키
            default_model: 기본 모델 이름
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = Anthropic(api_key=self.api_key)
        self.default_model = default_model

        # 빠른 모델 설정
        self.fast_model = os.getenv("CLAUDE_FAST_MODEL", "claude-3-5-haiku-20241022")

    def get_provider_name(self) -> str:
        return "claude"

    def get_default_model(self) -> str:
        return os.getenv("CLAUDE_CHAT_MODEL", self.default_model)

    def _convert_messages_to_claude_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """메시지를 Claude API 형식으로 변환"""
        claude_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Claude는 "user"와 "assistant"만 지원 (system은 별도 파라미터로)
            if role in ["user", "assistant"]:
                claude_messages.append({"role": role, "content": content})
            # system은 system 파라미터로 처리되므로 제외

        return claude_messages

    def _resolve_model(self, model: Optional[str], use_fast_model: bool) -> str:
        """
        모델 이름 결정: use_fast_model이 True면 빠른 모델, 아니면 model 또는 기본값
        """
        if use_fast_model:
            return self.fast_model
        return model or self.get_default_model()

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        **kwargs,
    ) -> str:
        """텍스트 생성"""
        claude_messages = self._convert_messages_to_claude_format(messages)
        model_name = self._resolve_model(model, use_fast_model)

        # max_tokens는 필수 파라미터
        max_tokens = kwargs.pop("max_tokens", 4096)

        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=claude_messages,
                system=system_instruction,
                **kwargs,
            )

            # Claude 응답은 content 리스트에서 텍스트 추출
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return ""
        except RateLimitError as e:
            raise ValueError(
                f"Claude API rate limit exceeded. Please try again later. Error: {str(e)}"
            ) from e
        except APIError as e:
            error_msg = str(e)
            if (
                "credit balance is too low" in error_msg
                or "credit" in error_msg.lower()
            ):
                raise ValueError(
                    f"Claude API credit balance is too low. Please upgrade your plan at https://console.anthropic.com/. Error: {error_msg}"
                ) from e
            raise ValueError(f"Claude API error: {error_msg}") from e
        except APIConnectionError as e:
            raise ValueError(
                f"Failed to connect to Claude API. Please check your network connection. Error: {str(e)}"
            ) from e

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        **kwargs,
    ) -> BaseModel:
        """
        Structured Output 생성 (Claude의 Structured Outputs 기능 사용)

        Claude의 beta API를 사용하여 Pydantic 모델에 맞는 구조화된 출력을 보장합니다.
        """
        claude_messages = self._convert_messages_to_claude_format(messages)
        model_name = self._resolve_model(model, use_fast_model)

        # max_tokens는 필수 파라미터
        max_tokens = kwargs.pop("max_tokens", 4096)

        try:
            # Claude의 beta.messages.parse() 사용
            # Pydantic 모델을 직접 전달하면 자동으로 스키마 변환 및 검증 수행
            response = self.client.beta.messages.parse(
                model=model_name,
                max_tokens=max_tokens,
                messages=claude_messages,
                system=system_instruction,
                betas=[
                    "structured-outputs-2025-11-13"
                ],  # Structured Outputs beta 기능 활성화
                output_format=response_schema,  # Pydantic 모델 직접 전달
                **kwargs,
            )

            # parse() 메서드는 자동으로 파싱된 결과를 반환
            # response.parsed_output에 이미 검증된 Pydantic 모델 인스턴스가 있음
            return response.parsed_output
        except RateLimitError as e:
            raise ValueError(
                f"Claude API rate limit exceeded. Please try again later. Error: {str(e)}"
            ) from e
        except APIError as e:
            error_msg = str(e)
            if (
                "credit balance is too low" in error_msg
                or "credit" in error_msg.lower()
            ):
                raise ValueError(
                    f"Claude API credit balance is too low. Please upgrade your plan at https://console.anthropic.com/. Error: {error_msg}"
                ) from e
            raise ValueError(f"Claude API error: {error_msg}") from e
        except APIConnectionError as e:
            raise ValueError(
                f"Failed to connect to Claude API. Please check your network connection. Error: {str(e)}"
            ) from e
