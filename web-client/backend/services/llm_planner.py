# Router가 수집한 컨텍스트로 리드 분류·액션·전달 플랜을 생성
# 최대 3가지의 응답 결과를 llm_evaluator로 전달

import json
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator
from .base_agent import BaseAgent
from prompts import (
    get_planner_role,
    get_planner_tool_guidelines,
    get_lead_classification_prompt,
)
from utils.logging_context import set_session_id

logger = logging.getLogger(__name__)


# Structured Outputs를 위한 Pydantic 모델 정의
class ToolCall(BaseModel):
    """Tool call 정보"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: str
    # arguments를 문자열로 받아서 strict 모드 요구사항 충족
    # 파싱 시 JSON으로 변환
    arguments: str

    def get_arguments_dict(self) -> Dict[str, Any]:
        """arguments 문자열을 Dict로 변환"""
        try:
            return json.loads(self.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}


class SpinStage(str, Enum):
    """SPIN 판매 단계"""

    SITUATION = "Situation"
    PROBLEM = "Problem"
    IMPLICATION = "Implication"
    NEED_PAYOFF = "Need-payoff"
    NONE = "해당없음"


class NudgeTechnique(str, Enum):
    """넛지 기법"""

    LOSS_AVERSION = "손실회피"
    FRAMING = "프레이밍"
    ANCHORING = "앵커링"
    DEFAULT_EFFECT = "기본값효과"
    SCARCITY = "희소성"
    DECOY_EFFECT = "유인 효과"
    NONE = "해당없음"


class LeadStatus(str, Enum):
    """리드 상태"""

    COLD_LEAD = "Cold Lead"
    WARM_LEAD = "Warm Lead"
    HOT_LEAD = "Hot Lead"
    CONVERTED = "Converted"
    LOST = "Lost"


class Step1Classification(BaseModel):
    """STEP 1: 리드 상태 분류"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    lead_status: LeadStatus
    classification_reason: str

    @field_validator("lead_status", mode="before")
    @classmethod
    def normalize_lead_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class Reasoning(BaseModel):
    """STEP 1: 리드 상태 분류 (액션은 플랜별 intent)"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    step1_classification: Step1Classification


class Plan(BaseModel):
    """STEP 2+3: 세일즈 액션 + 전달 플랜"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    plan_id: str  # "A", "B", "C"
    intent: str  # "SPIN 질문", "차량 추천/비교", "구매장벽 해소"
    strategic_approach: str
    spin_stage: SpinStage
    nudge_technique: NudgeTechnique
    why_this_strategy: str
    dialogue: str
    tool_calls: List[ToolCall] = []

    # Claude 등 일부 LLM이 Enum 값을 앞에 공백을 붙여 반환하는 경우가 있어
    # Pydantic enum 검증 전에 공백을 제거하여 ValidationError를 방지한다.
    @field_validator("spin_stage", mode="before")
    @classmethod
    def normalize_spin_stage(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("nudge_technique", mode="before")
    @classmethod
    def normalize_nudge_technique(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class PlannerResponse(BaseModel):
    """Planner 응답 구조 (STEP 1 리드 + 플랜별 STEP 2/3)"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    reasoning: Reasoning
    plans: List[Plan]  # 최대 3개 (Plan A, B, C)


class LLMPlanner(BaseAgent):
    """사용자 질문에 대한 의도 분석 및 최대 3개 응답 시나리오 생성"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(api_key)
        self.last_response_id: Optional[str] = None  # previous_response_id 체인용

        # 프롬프트 캐싱
        self._cached_base_prompt = None
        self._cached_planner_role = None
        self._cached_planner_tool_guidelines = None

    def _get_cached_base_prompts(self):
        """캐시된 기본 프롬프트 반환"""
        # 캐시가 없으면 로드
        if self._cached_base_prompt is None:
            self._cached_base_prompt = self.get_common_sales_knowledge()
            self._cached_planner_role = get_planner_role()
            self._cached_planner_tool_guidelines = get_planner_tool_guidelines()
            logger.debug("[LLMPlanner] Prompts cached successfully")

        return (
            self._cached_base_prompt,
            self._cached_planner_role,
            self._cached_planner_tool_guidelines,
        )

    def get_system_prompt(
        self,
        rag_context: Optional[str] = None,
        vehicle_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_info: Optional[Dict[str, Any]] = None,
        guess_sections: Optional[List[str]] = None,
    ) -> str:
        """Planner Agent 전용 System Prompt 생성"""
        (
            base_prompt,
            planner_role,
            planner_tool_guidelines,
        ) = self._get_cached_base_prompts()
        lead_classification_guide = get_lead_classification_prompt()

        # RAG 검색 결과 추가
        rag_section = ""
        if (
            rag_context
            and rag_context.strip()
            and not rag_context.startswith("No information found")
            and not rag_context.startswith("No relevant information found")
        ):
            rag_section = f"""
              ============================================================
              [RAG 검색 결과 - 반드시 활용하세요]
              ============================================================

              {rag_context}

              [CRITICAL 지침 - 반드시 따라야 합니다]
              1. 위 RAG 검색 결과에 포함된 차량 정보를 반드시 활용하여 dialogue를 생성하세요.
              2. "차량 정보를 확인해볼게요", "정보를 찾아보겠습니다", "다음 세 가지 중고차를 추천드립니다" 같은 모호한 응답만으로는 절대 안 됩니다.
              3. RAG 검색 결과의 실제 내용(모델명, 연식, 색상, 가격, 사양, 옵션 등)을 dialogue에 구체적으로 포함해야 합니다.
              4. 추천 요청의 경우, RAG 검색 결과에 나온 각 차량의 구체적인 정보를 dialogue에 포함하세요.
                예: "다음 세 가지 중고차를 추천드립니다. 첫 번째는 [모델명], [연식], [주행거리], [가격], [주요 사양]..."
              5. 사용자가 "현재 차량", "이 차량", "자세히 알려줘" 등으로 요청하면, RAG 검색 결과의 모든 주요 정보를 포함한 상세한 응답을 생성하세요.
              6. RAG 검색 결과에 없는 정보는 추측하지 마세요. 오직 RAG 검색 결과에 있는 정보만 사용하세요.
              7. 사용자가 한국어로 질문하면 반드시 한국어로 응답하세요.
              8. dialogue는 RAG 검색 결과를 기반으로 구체적이고 상세한 정보를 제공해야 합니다 (300-500자 권장, 최대 800자).
              9. dialogue가 모호하거나 구체적인 정보가 없으면 시나리오가 무효화됩니다.
              """

        # 컨텍스트 섹션 추가
        context_section = self.build_context_section(vehicle_id, conversation_history)

        tool_definitions_section = self._format_tool_definitions(
            self.get_planner_tool_definitions()
        )

        # 사용자 정보 섹션 추가
        user_info_section = ""
        if user_info:
            user_info_parts = []
            if user_info.get("purchase_purpose"):
                purposes = ", ".join(user_info["purchase_purpose"])
                user_info_parts.append(f"구매 목적: {purposes}")
            if user_info.get("preferred_car_type"):
                user_info_parts.append(f"선호 차종: {user_info['preferred_car_type']}")
            if user_info.get("budget_krw"):
                user_info_parts.append(f"예산: {user_info['budget_krw']}만원")

            if user_info_parts:
                user_info_section = f"""
                ============================================================
                [수집된 사용자 정보]
                ============================================================
                {chr(10).join(user_info_parts)}
                
                중요: 위 정보를 반드시 고려하여 응답을 생성하세요.
                - 구매 목적에 맞는 차량 추천
                - 선호 차종에 맞는 차량 제안
                - 예산 범위 내 차량 제안
                ============================================================
                """

        # 추정된 섹션 정보 추가
        guess_sections_section = ""
        if guess_sections:
            guess_sections_section = f"""
                ============================================================
                [필요한 차량 정보 섹션]
                ============================================================
                다음 섹션들의 정보가 특히 중요합니다:
                {', '.join(guess_sections)}
                
                RAG 검색 결과에서 위 섹션 정보를 우선적으로 활용하세요.
                ============================================================
                """

        prompt_parts = [
            base_prompt,
            planner_role,
            lead_classification_guide,
            planner_tool_guidelines,
            tool_definitions_section,
            user_info_section,
            guess_sections_section,
            rag_section,
            context_section,
        ]

        return "\n\n".join(filter(None, prompt_parts))  # 빈 문자열 제거

    def plan(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        vehicle_id: Optional[str] = None,
        session_id: Optional[str] = None,
        router_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Router가 준 컨텍스트로 최대 3개의 응답 플랜 생성.
        검색은 하지 않는다.
        """
        set_session_id(session_id or "default")
        ctx = router_context or {}
        rag_context = ctx.get("rag_context")
        guess_sections = ctx.get("guess_sections")
        user_info = ctx.get("user_info") or self.get_user_info(session_id)

        logger.info("=" * 80)
        logger.info("[Planner] 시나리오 생성 시작 (검색 없음)")
        logger.info(f"  - 사용자 메시지: {message}")
        logger.info(f"  - Vehicle ID: {vehicle_id if vehicle_id else '없음'}")
        logger.info(f"  - RAG 길이: {len(rag_context) if rag_context else 0}")
        logger.info("=" * 80)

        system_prompt = self.get_system_prompt(
            rag_context=rag_context,
            vehicle_id=vehicle_id,
            conversation_history=conversation_history,
            user_info=user_info,
            guess_sections=guess_sections,
        )

        # 입력 메시지 구성 (role 기반 메시지 배열 - 가이드 권장 방식)
        # 요약본 + 최근 메시지 조합 사용
        effective_history = self.get_effective_history(conversation_history, session_id)
        input_messages = []
        if effective_history:
            input_messages.extend(effective_history)
        input_messages.append({"role": "user", "content": message})

        try:
            planner_response = self.llm_client.generate_structured(
                messages=input_messages,
                response_schema=PlannerResponse,
                system_instruction=system_prompt,
                session_id=session_id,
            )

            step1 = planner_response.reasoning.step1_classification
            logger.info(
                f"[Planner] Step1 리드 상태: {step1.lead_status.value}, "
                f"이유: {step1.classification_reason}"
            )

            # PlannerResponse의 plans를 딕셔너리로 변환
            result = []

            for plan in planner_response.plans[:3]:  # 최대 3개 (Plan A, B, C)
                logger.info(
                    f"[Planner] 플랜 {plan.plan_id} 생성: intent={getattr(plan, 'intent', None)}, "
                    f"strategic_approach={plan.strategic_approach}, "
                    f"spin_stage={plan.spin_stage.value}, nudge_technique={plan.nudge_technique.value}, "
                    f"tool_calls={len(plan.tool_calls)}개"
                )
                logger.info(
                    f"[Planner] 플랜 {plan.plan_id} 상세 - dialogue: {plan.dialogue[:200]}..., "
                    f"why_this_strategy: {plan.why_this_strategy[:200]}..."
                )

                plan_dict = plan.model_dump(mode="python")

                # tool_calls의 arguments를 Dict로 변환
                if "tool_calls" in plan_dict and plan_dict["tool_calls"]:
                    processed_tool_calls = []
                    for tool_call in plan_dict["tool_calls"]:
                        # Pydantic 모델인 경우 한 번에 처리
                        if hasattr(tool_call, "model_dump"):
                            tool_call = tool_call.model_dump(mode="python")
                        elif not isinstance(tool_call, dict):
                            continue

                        # arguments JSON 파싱
                        if isinstance(tool_call.get("arguments"), str):
                            args_str = tool_call.get("arguments")
                            tool_name = tool_call.get("name")
                            try:
                                tool_call["arguments"] = json.loads(args_str)
                            except (json.JSONDecodeError, TypeError) as e:
                                # JSON 파싱 실패 시 상세 로깅
                                error_pos = getattr(e, "pos", None) or 0
                                logger.warning(
                                    f"JSON 파싱 실패: {tool_name}, error={e}"
                                )
                                # 에러 위치 주변의 JSON 문자열 추출 (디버깅용)
                                if error_pos and error_pos < len(args_str):
                                    start = max(0, error_pos - 100)
                                    end = min(len(args_str), error_pos + 100)
                                    logger.warning(
                                        f"JSON 파싱 실패 위치 주변 (char {start}-{end}): {repr(args_str[start:end])}"
                                    )
                                # 전체 JSON 문자열 길이와 일부 내용 로깅 (너무 길면 처음/끝만)
                                if len(args_str) > 500:
                                    logger.warning(
                                        f"JSON 전체 길이: {len(args_str)}자, 처음 500자: {repr(args_str[:500])}"
                                    )
                                    logger.warning(
                                        f"JSON 마지막 200자: {repr(args_str[-200:])}"
                                    )
                                else:
                                    logger.warning(f"JSON 전체 내용: {repr(args_str)}")
                                tool_call["arguments"] = {}

                        processed_tool_calls.append(tool_call)
                    plan_dict["tool_calls"] = [
                        tc
                        for tc in processed_tool_calls
                        if tc.get("name") != "searchVehicleDatabase"
                    ]

                plan_dict["rag_context"] = rag_context
                plan_dict["evidence"] = ctx.get("evidence") or []
                plan_dict["lead_status"] = step1.lead_status.value

                result.append(plan_dict)

            # previous_response_id 저장 (다음 호출 시 체인 연결용 - 선택적)
            # Gemini API는 response.id가 없을 수 있으므로 체크
            if hasattr(planner_response, "id"):
                self.last_response_id = planner_response.id
            elif (
                hasattr(planner_response, "candidates") and planner_response.candidates
            ):
                pass

            # 최종 결과 요약
            logger.info("=" * 80)
            logger.info(f"[Planner] ✅ 시나리오 생성 완료 (총 {len(result)}개)")
            for idx, plan in enumerate(result):
                logger.info(
                    f"  [{idx+1}] Plan {plan.get('plan_id')}: {plan.get('strategic_approach')}"
                )
                logger.info(f"      - Tool Calls: {len(plan.get('tool_calls', []))}개")
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"Planner 오류 발생: {e}", exc_info=True)
            return self._create_error_plan(str(e), rag_context)

    def _format_tool_definitions(self, tools: List[Dict[str, Any]]) -> str:
        """Tool 정의를 System Prompt에 포함하기 위한 형식으로 변환"""
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # 파라미터 요약
            param_summary = []
            if "properties" in parameters:
                for param_name, param_def in parameters["properties"].items():
                    param_type = param_def.get("type", "unknown")
                    param_desc = param_def.get("description", "")
                    required = param_name in parameters.get("required", [])
                    req_marker = " (필수)" if required else " (선택)"
                    param_summary.append(
                        f"  - {param_name} ({param_type}){req_marker}: {param_desc}"
                    )

            tool_descriptions.append(
                f"""
              [{name}]
              설명: {description}
              파라미터:
              {chr(10).join(param_summary) if param_summary else "  없음"}
              """
            )

        return f"""
          ============================================================
          [사용 가능한 Tool 정의]
          ============================================================
          다음 tool들을 tool_calls에 포함할 수 있습니다:

          {chr(10).join(tool_descriptions)}

          [중요]
          - tool_calls는 위 정의에 맞는 형식으로 생성해야 합니다.
          - arguments는 JSON 문자열로 제공되며, 위 파라미터 정의에 맞춰야 합니다.
          - 필수 파라미터는 반드시 포함해야 합니다.
          """

    def _create_error_plan(
        self, error_msg: str, rag_context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """에러 발생 시 기본 플랜 생성"""
        return [
            {
                "plan_id": "A",
                "intent": "SPIN 질문",
                "strategic_approach": "오류_처리",
                "spin_stage": SpinStage.NONE.value,
                "nudge_technique": NudgeTechnique.NONE.value,
                "why_this_strategy": f"API 호출 오류: {error_msg}",
                "dialogue": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "tool_calls": [],
                "rag_context": rag_context,
            }
        ]

