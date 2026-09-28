"""
LLM Client Interface - 공통 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel


class LLMClient(ABC):
    """LLM 클라이언트 공통 인터페이스"""

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        **kwargs
    ) -> str:
        """
        텍스트 생성 (간단한 텍스트 응답)

        Args:
            messages: 대화 메시지 리스트 [{"role": "user", "content": "..."}, ...]
            system_instruction: 시스템 프롬프트
            model: 모델 이름 (None이면 기본값 사용)
            use_fast_model: True일 경우 provider별 빠른 모델 사용 (model 파라미터보다 우선)
            **kwargs: 추가 파라미터

        Returns:
            생성된 텍스트
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        **kwargs
    ) -> BaseModel:
        """
        Structured Output 생성 (Pydantic 모델로 응답)

        Args:
            messages: 대화 메시지 리스트
            response_schema: Pydantic 모델 클래스
            system_instruction: 시스템 프롬프트
            model: 모델 이름
            use_fast_model: True일 경우 provider별 빠른 모델 사용 (model 파라미터보다 우선)
            **kwargs: 추가 파라미터

        Returns:
            파싱된 Pydantic 모델 인스턴스
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """기본 모델 이름 반환"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """프로바이더 이름 반환 (예: "openai", "gemini", "claude")"""
        pass
