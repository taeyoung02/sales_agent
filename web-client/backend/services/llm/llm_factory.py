"""
LLM Client Factory - Provider 선택 및 인스턴스 생성
"""

import os
from enum import Enum
from typing import Optional

from .llm_client_interface import LLMClient
from .openai_client import OpenAILLMClient
from .gemini_client import GeminiLLMClient
from .claude_client import ClaudeLLMClient


class LLMProvider(str, Enum):
    """지원하는 LLM 프로바이더"""

    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


class LLMClientFactory:
    """LLM 클라이언트 팩토리"""

    @staticmethod
    def create_client(
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> LLMClient:
        """
        LLM 클라이언트 생성

        Args:
            provider: 프로바이더 (None이면 환경 변수에서 읽음)
            api_key: API 키 (None이면 환경 변수에서 읽음)
            default_model: 기본 모델 이름

        Returns:
            LLMClient 인스턴스

        Raises:
            ValueError: 지원하지 않는 프로바이더 또는 API 키 누락
        """
        # 프로바이더 결정: 명시적 지정 > 환경 변수 > 기본값 (gemini)
        if provider is None:
            provider_str = os.getenv("LLM_PROVIDER", "gemini").lower()
            try:
                provider = LLMProvider(provider_str)
            except ValueError:
                raise ValueError(
                    f"Unsupported LLM provider: {provider_str}. "
                    f"Supported: {[p.value for p in LLMProvider]}"
                )

        # 클라이언트 생성
        if provider == LLMProvider.OPENAI:
            return OpenAILLMClient(
                api_key=api_key, default_model=default_model or "gpt-4o"
            )
        elif provider == LLMProvider.GEMINI:
            return GeminiLLMClient(
                api_key=api_key, default_model=default_model or "gemini-2.5-flash"
            )
        elif provider == LLMProvider.CLAUDE:
            return ClaudeLLMClient(
                api_key=api_key,
                default_model=default_model or "claude-sonnet-4-20250514",
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_provider_from_env() -> LLMProvider:
        """환경 변수에서 프로바이더 읽기"""
        provider_str = os.getenv("LLM_PROVIDER", "gemini").lower()
        try:
            return LLMProvider(provider_str)
        except ValueError:
            return LLMProvider.GEMINI  # 기본값
