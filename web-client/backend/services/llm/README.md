# LLM Provider Abstraction Layer

OpenAI와 Gemini 두 LLM API를 공통 인터페이스로 관리하는 추상화 레이어입니다.

## 설계 패턴

- **Strategy Pattern**: 각 LLM 프로바이더를 전략으로 구현
- **Factory Pattern**: 프로바이더 선택 및 인스턴스 생성
- **Adapter Pattern**: 각 API의 차이점을 공통 인터페이스로 통일

## 구조

```
llm/
├── __init__.py              # 공개 API
├── llm_client_interface.py  # 추상 기본 클래스 (공통 인터페이스)
├── openai_client.py         # OpenAI 구현
├── gemini_client.py         # Gemini 구현
├── llm_factory.py           # 팩토리 (프로바이더 선택)
└── README.md                # 문서
```

## 사용 방법

### 1. 기본 사용 (환경 변수에서 프로바이더 읽기)

```python
from services.llm import LLMClientFactory

# 환경 변수 LLM_PROVIDER 설정 (openai 또는 gemini)
# export LLM_PROVIDER=gemini  또는  export LLM_PROVIDER=openai

llm_client = LLMClientFactory.create_client()

# 텍스트 생성
response = llm_client.generate_text(
    messages=[{"role": "user", "content": "안녕하세요"}],
    system_instruction="당신은 친절한 어시스턴트입니다."
)

# Structured Output
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    score: int

result = llm_client.generate_structured(
    messages=[{"role": "user", "content": "점수 5점으로 답변해주세요"}],
    response_schema=Response,
    system_instruction="JSON 형식으로 응답하세요."
)

print(result.answer)  # "안녕하세요"
print(result.score)   # 5
```

### 2. 명시적으로 프로바이더 지정

```python
from services.llm import LLMClientFactory, LLMProvider

# OpenAI 사용
openai_client = LLMClientFactory.create_client(provider=LLMProvider.OPENAI)

# Gemini 사용
gemini_client = LLMClientFactory.create_client(provider=LLMProvider.GEMINI)
```

### 3. BaseAgent에서 사용

```python
from services.llm import LLMClientFactory
from services.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        # LLM 클라이언트는 공통 인터페이스 사용
        self.llm_client = LLMClientFactory.create_client()
    
    def my_method(self):
        # 프로바이더에 관계없이 동일한 방식으로 사용
        response = self.llm_client.generate_text(
            messages=[{"role": "user", "content": "..."}],
            system_instruction="..."
        )
```

## 환경 변수

- `LLM_PROVIDER`: 사용할 프로바이더 (`openai` 또는 `gemini`, 기본값: `gemini`)
- `OPENAI_API_KEY`: OpenAI API 키
- `GEMINI_API_KEY`: Gemini API 키
- `OPENAI_CHAT_MODEL`: OpenAI 기본 모델 (기본값: `gpt-4o`)
- `GEMINI_CHAT_MODEL`: Gemini 기본 모델 (기본값: `gemini-2.5-flash`)

## 공통 인터페이스

모든 LLM 클라이언트는 다음 메서드를 구현합니다:

- `generate_text()`: 간단한 텍스트 생성
- `generate_structured()`: Pydantic 모델로 구조화된 응답 생성
- `get_default_model()`: 기본 모델 이름
- `get_provider_name()`: 프로바이더 이름

## 장점

1. **프로바이더 전환 용이**: 환경 변수 하나만 변경하면 전체 시스템이 다른 프로바이더 사용
2. **코드 일관성**: 프로바이더에 관계없이 동일한 코드 패턴
3. **확장성**: 새로운 프로바이더 추가가 쉬움 (LLMClient 상속 후 팩토리에 등록)
4. **테스트 용이**: Mock 객체로 쉽게 테스트 가능
