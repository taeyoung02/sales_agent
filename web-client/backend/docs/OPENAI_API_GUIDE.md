# OpenAI API (Python) 코드 작성 가이드라인

> 참고 문서: Text generation, Structured Outputs, Function calling

---

## 0) 기본 전제: Python SDK + Responses API를 기준으로 잡기

텍스트/구조화/툴 호출 모두 **Responses API** 흐름으로 통일하면, "대화 상태 관리 + 도구 호출 + 구조화 출력"을 한 가지 인터페이스로 묶어 구현하기 쉽습니다.

---

## 1) 설치 & 인증(필수)

### 1.1 설치

```bash
pip install openai
```

### 1.2 API Key 설정

```bash
export OPENAI_API_KEY="..."
```

또는 환경 변수 파일(`.env`) 사용:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### 1.3 클라이언트 생성

```python
from openai import OpenAI

client = OpenAI()
# 또는 명시적으로 API 키 전달
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

---

## 2) Text: 가장 기본적인 텍스트 생성 패턴

### 2.1 단일 입력 → 단일 텍스트 출력

```python
from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o",
    input="한 문장으로 유니콘 동화를 써줘.",
)

print(resp.output_text)
```

### 2.2 `output_text`를 기본값으로 쓰는 이유

Responses의 `output` 배열에는 텍스트 외에도 **툴 호출, reasoning 아이템 등 여러 항목**이 섞여 들어갈 수 있어서, `output[0].content[0].text` 같은 "고정 인덱스 접근"은 안전하지 않습니다.

SDK가 제공하는 `output_text`는 모델이 만든 모든 텍스트를 합쳐서 꺼내는 **안전한 단축키**로 쓰기 좋습니다.

### 2.3 "지시문(instructions)"과 "메시지(role)"를 섞어 쓰는 규칙

* **상위 정책/톤/제약**: `instructions`로 (우선순위가 높음)
* **대화 맥락/콘텐츠**: `input`에 role 기반 메시지로

```python
resp = client.responses.create(
    model="gpt-4o",
    instructions="너는 간결한 기술 문서 작성자야. 불필요한 수식어는 빼.",
    input=[
        {"role": "user", "content": "Redis와 Memcached 차이를 5줄로 정리해줘."}
    ],
)

print(resp.output_text)
```

---

## 3) Conversation State: 대화 상태(컨텍스트) 전달 방법 3가지

### 3.1 (가장 단순) 매 요청마다 히스토리를 직접 포함하기

`user`/`assistant`를 번갈아 넣어 "지금까지 대화"를 한 번에 전달할 수 있습니다.

```python
resp = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "user", "content": "knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
)

print(resp.output_text)
```

### 3.2 (추천) `previous_response_id`로 체인 연결하기

이 방식은 "이전 응답의 문맥"을 자동으로 이어받아 **멀티턴을 깔끔하게 체인**으로 구성합니다.

```python
first = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
)

second = client.responses.create(
    model="gpt-4o-mini",
    previous_response_id=first.id,
    input=[{"role": "user", "content": "explain why this is funny."}],
)

print(second.output_text)
```

### 3.3 (세션/디바이스/잡 단위) Conversations API로 "대화 객체"를 유지하기

`conversation="conv_..."` 형태로 장기 식별자를 가진 대화 컨테이너를 붙여서 지속 상태를 관리합니다.

```python
# 대화 생성
conversation = client.conversations.create()

# 대화에 메시지 추가
resp = client.responses.create(
    model="gpt-4o",
    conversation=conversation.id,
    input="안녕하세요",
)

# 같은 대화에 계속 추가
resp2 = client.responses.create(
    model="gpt-4o",
    conversation=conversation.id,
    input="이전 대화를 기억하나요?",
)
```

---

## 4) Structured Outputs: "정해진 스키마로" 안전하게 JSON 뽑기

### 4.1 언제 쓰나?

* **UI/DB/파이프라인**이 "정확한 타입/필드"를 필요로 할 때
* "검증/리트라이 프롬프트"를 최소화하고 싶을 때

Structured Outputs는 **타입 안정성**, **명시적 거절(Refusal) 감지**, **프롬프트 단순화**에 유리합니다.

### 4.2 Python(Pydantic)으로 스키마 정의 + `responses.parse()` 사용

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

resp = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    text_format=CalendarEvent,
)

event: CalendarEvent = resp.output_parsed
print(event)
```

### 4.3 모델 호환성 주의

Structured Outputs는 "최신 대형 모델들(GPT-4o부터)"에서 지원되는 형태로 안내됩니다.

### 4.4 `text.format` vs Function calling(툴 호출) 중 무엇을 선택?

* **단순 추출/정규화/분류 결과를 JSON으로**: `text.format`(Structured Outputs) 쪽이 직관적
* **모델이 "내 시스템의 기능(조회/결제/예약/DB)"을 호출해야**: Function calling이 맞음

---

## 5) Function Calling: 모델이 "내 코드/시스템 기능"을 호출하게 만들기

### 5.1 핵심 개념

* 요청에 `tools=[...]`로 **함수 스키마(JSON Schema)**를 넘기면
* 모델이 필요할 때 `output`에 `type="function_call"`을 반환
* 서버가 실제 함수를 실행한 뒤, 결과를 `type="function_call_output"`으로 다시 모델에 전달

(모델은 0개/1개/여러 개 툴 호출을 할 수 있으니 "여러 개 가능"을 기본 가정)

### 5.2 함수 스키마 작성 규칙(중요)

* `parameters`는 JSON Schema
* **strict 모드 권장**: `strict=True`로 스키마 준수도를 올림
* strict 요구사항:
  * 각 object에 `additionalProperties: false`
  * `properties`의 모든 필드를 `required`에 포함 (선택 필드는 `type: ["...", "null"]`로 표현)

### 5.3 End-to-End 예시(가장 표준적인 구현)

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {"type": "string", "description": "e.g. Taurus, Aquarius"},
            },
            "required": ["sign"],
            "additionalProperties": False,
        },
    }
]

def get_horoscope(sign: str) -> str:
    return f"{sign}: Next Tuesday you will befriend a baby otter."

input_list = [{"role": "user", "content": "What is my horoscope? I am an Aquarius."}]

# 1) 모델에게 툴을 포함해 질문
resp = client.responses.create(
    model="gpt-4o",
    tools=tools,
    input=input_list,
)

# 2) 다음 턴으로 넘기기 위해 output을 input에 누적
input_list += resp.output

# 3) function_call 처리
for item in resp.output:
    if item.type != "function_call":
        continue
    if item.name == "get_horoscope":
        args = json.loads(item.arguments)
        result = get_horoscope(**args)

        input_list.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps({"horoscope": result}),
        })

# 4) 결과를 포함해 다시 모델 호출 → 최종 답변 생성
final = client.responses.create(
    model="gpt-4o",
    instructions="Respond only with a horoscope generated by a tool.",
    tools=tools,
    input=input_list,
)

print(final.output_text)
```

> 참고: reasoning 모델(GPT-5, o4-mini 등)에서 툴 호출이 섞인 응답을 체인할 때, 응답에 포함된 reasoning 아이템도 함께 다시 전달해야 한다는 주의가 있습니다. (툴 결과만 보내지 말고, 모델이 반환한 output 아이템들을 누적하는 패턴을 쓰는 이유)

### 5.4 Tool 선택 제어(`tool_choice`) & 병렬 호출 제어

* **기본**: 모델이 알아서 호출(`tool_choice: "auto"`)
* **반드시 호출하게 강제**: `"required"`
* **특정 함수만 강제**: `{"type":"function","name":"..."}`
* **병렬 호출 차단**: `parallel_tool_calls=false`로 "0 또는 1개만" 호출되게

```python
# 특정 함수만 호출하게 강제
resp = client.responses.create(
    model="gpt-4o",
    tools=tools,
    tool_choice={"type": "function", "name": "get_horoscope"},
    input="What's my horoscope?",
)

# 병렬 호출 비활성화 (순차적으로만 호출)
resp = client.responses.create(
    model="gpt-4o",
    tools=tools,
    parallel_tool_calls=False,
    input="Check multiple things",
)
```

---

## 6) 구현 체크리스트(실무용)

### 6.1 공통

* [ ] 프로덕션은 모델 스냅샷을 "핀"해서 행동 변동을 줄이기
* [ ] 텍스트는 `resp.output_text`를 기본으로 사용 (인덱스 가정 금지)
* [ ] 대화형이면 `previous_response_id`로 체인 (또는 Conversations API)

### 6.2 Structured Outputs

* [ ] 파이프라인 경계(저장/전송/API 응답)에서는 `responses.parse()` + Pydantic으로 타입 고정
* [ ] "기능 실행"이 필요 없으면 Function calling 대신 `text.format` 우선 고려

### 6.3 Function calling

* [ ] `strict=True` + JSON Schema 요구사항 준수
* [ ] function_call은 "여러 개" 가능성을 기본 가정하고 루프 처리
* [ ] tool 결과는 반드시 문자열(`output`)로 반환 (형식은 JSON/텍스트 등 자유)

---

## 7) 추천 설계 패턴(요약)

* **"그냥 답변 생성"**: `responses.create()` + `output_text`
* **"대화"**: `previous_response_id` 체인 (짧고 깔끔) 또는 Conversations API (장기 세션)
* **"정확한 JSON 결과가 필요"**: `responses.parse()` (Pydantic)
* **"내 시스템 기능을 호출"**: Function calling + strict + (call_id → function_call_output) 루프

---

## 8) 실제 프로젝트 적용 예시

### 8.1 BaseAgent 패턴 (현재 프로젝트 구조)

```python
from openai import OpenAI
from typing import Optional, List, Dict, Any
import os

class BaseAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
    
    def _extract_content_from_response(self, response) -> str:
        """Responses API 응답에서 텍스트 콘텐츠 추출"""
        if hasattr(response, "output_text"):
            return response.output_text
        
        # Fallback: output 배열에서 추출
        if hasattr(response, "output") and response.output:
            output = response.output
            if isinstance(output, list) and len(output) > 0:
                first_message = output[0]
                if hasattr(first_message, "text"):
                    return first_message.text
                elif hasattr(first_message, "content"):
                    content = first_message.content
                    if isinstance(content, list):
                        return "".join(str(item) for item in content)
                    return str(content)
        
        return str(response)
```

### 8.2 대화 히스토리 관리 패턴

```python
class ConversationManager:
    def __init__(self, client: OpenAI):
        self.client = client
        self.conversation_history: List[Dict[str, str]] = []
        self.last_response_id: Optional[str] = None
    
    def add_user_message(self, message: str):
        self.conversation_history.append({"role": "user", "content": message})
    
    def get_response(self, instructions: str, model: str = "gpt-4o") -> str:
        if self.last_response_id:
            # 이전 응답과 체인 연결
            resp = self.client.responses.create(
                model=model,
                instructions=instructions,
                previous_response_id=self.last_response_id,
                input=[self.conversation_history[-1]],
            )
        else:
            # 첫 번째 요청
            resp = self.client.responses.create(
                model=model,
                instructions=instructions,
                input=self.conversation_history,
            )
        
        self.last_response_id = resp.id
        self.conversation_history.append({
            "role": "assistant",
            "content": resp.output_text
        })
        
        return resp.output_text
```

### 8.3 Function Calling 패턴 (현재 프로젝트 스타일)

```python
class ToolExecutor:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def execute_with_tools(
        self,
        message: str,
        tools: List[Dict],
        instructions: str,
        max_iterations: int = 5
    ) -> str:
        input_list = [{"role": "user", "content": message}]
        
        for _ in range(max_iterations):
            resp = self.client.responses.create(
                model="gpt-4o",
                instructions=instructions,
                tools=tools,
                input=input_list,
            )
            
            # output을 다음 입력에 추가
            input_list += resp.output
            
            # function_call이 있는지 확인
            has_function_call = any(
                item.type == "function_call" for item in resp.output
            )
            
            if not has_function_call:
                # 최종 응답 반환
                return resp.output_text
            
            # function_call 처리
            for item in resp.output:
                if item.type != "function_call":
                    continue
                
                # 실제 함수 실행
                result = self._execute_function(item.name, item.arguments)
                
                # 결과를 input에 추가
                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                })
        
        return resp.output_text
    
    def _execute_function(self, name: str, arguments: str) -> Dict:
        # 실제 함수 실행 로직
        args = json.loads(arguments)
        # ... 함수 실행 ...
        return {"result": "..."}
```

---

## 9) 에러 처리 및 모범 사례

### 9.1 에러 처리

```python
from openai import APIError, RateLimitError

try:
    resp = client.responses.create(
        model="gpt-4o",
        input="Hello",
    )
except RateLimitError:
    # Rate limit 처리
    print("Rate limit exceeded. Please retry later.")
except APIError as e:
    # 일반 API 에러
    print(f"API Error: {e}")
except Exception as e:
    # 기타 에러
    print(f"Unexpected error: {e}")
```

### 9.2 타임아웃 설정

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,  # 30초 타임아웃
    max_retries=3,  # 최대 3번 재시도
)
```

### 9.3 모델 버전 고정 (프로덕션)

```python
# 스냅샷 버전 사용 (행동 변동 최소화)
resp = client.responses.create(
    model="gpt-4o-2024-08-06",  # 특정 날짜 스냅샷
    input="Hello",
)
```

---

## 10) 참고 자료

* [OpenAI Platform - Text generation](https://platform.openai.com/docs/guides/text)
* [OpenAI Platform - Conversation state](https://platform.openai.com/docs/guides/conversation-state)
* [OpenAI Platform - Structured outputs](https://platform.openai.com/docs/guides/structured-outputs)
* [OpenAI Platform - Function calling](https://platform.openai.com/docs/guides/function-calling)
* [OpenAI Python SDK Documentation](https://github.com/openai/openai-python)

