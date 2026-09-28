# llm_planner 에이전트가 도출한 출력 결과에 대한 평가
# RAG 근거 대비 faithfulness / factuality를 검증하고 최종 1개를 반환

import json
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict
from .base_agent import BaseAgent
from prompts import get_evaluator_role
from utils.logging_context import set_session_id

logger = logging.getLogger(__name__)


class DataIntegrityScores(BaseModel):
    """데이터 무결성 (faithfulness / factuality)"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    faithfulness: int
    factuality: int
    소계: int


class IntentAlignmentScores(BaseModel):
    """인텐트 정합성"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    비즈니스_전략_준수: int
    고객_동기_강화: int
    소계: int


class PersonalizedPersuasionScores(BaseModel):
    """개인화 설득 효용성"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    유저_성향_적합성: int
    넛지_전략_효과: int
    소계: int


class PlanEvaluationScores(BaseModel):
    """플랜별 평가 점수"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    데이터_무결성: DataIntegrityScores
    인텐트_정합성: IntentAlignmentScores
    개인화_설득_효용성: PersonalizedPersuasionScores
    총점: int


class PlanEvaluation(BaseModel):
    """단일 플랜 평가"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    plan_id: str  # "A", "B", "C"
    scores: PlanEvaluationScores
    평가_코멘트: str


class FinalOutput(BaseModel):
    """최종 출력"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    dialogue: str
    spin_stage: str
    nudge_technique: str


class EvaluatorResponse(BaseModel):
    """Evaluator 응답 구조"""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    evaluations: List[PlanEvaluation]
    selected_plan: str  # "A", "B"
    selection_reasoning: str
    final_output: FinalOutput


class LLMEvaluator(BaseAgent):
    """llm_planner 에이전트가 도출한 출력 결과 (최대 2개)에 대한 평가를 진행하는 에이전트"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(api_key)

    def get_system_prompt(
        self,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Evaluator Agent 전용 System Prompt 생성
        공통 세일즈 지식 + Evaluator 역할 + 사용자 정보
        """

        # 공통 세일즈 지식
        base_prompt = self.get_common_sales_knowledge()

        # 공통 Tool 가이드라인 (평가 시 tool 이해 필요)
        common_tool_guidelines = self.get_common_tool_guidelines()

        # Evaluator 역할 추가
        evaluator_role = get_evaluator_role()

        # 공유된 사용자 정보 추가
        user_info = self.get_user_info(session_id)
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
        
        중요: 위 정보를 반드시 고려하여 플랜을 평가하세요.
        - 구매 목적에 맞는 플랜 우선 선택
        - 선호 차종에 맞는 플랜 우선 선택
        - 예산 범위 내 추천이 포함된 플랜 우선
        ============================================================
        """

        # 대화 히스토리는 role 기반 메시지로 전달하므로 프롬프트에 포함하지 않음
        return (
            base_prompt
            + "\n\n"
            + common_tool_guidelines
            + "\n\n"
            + evaluator_role
            + "\n\n"
            + user_info_section
        )

    def evaluate(
        self,
        plans: List[Dict[str, Any]],
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        플랜들을 평가하여 최종 1개 선택

        Args:
            plans: Planner가 생성한 플랜 리스트 (최대 2개: Plan A, Plan B)
            user_message: 원본 사용자 메시지
            conversation_history: 대화 히스토리

        Returns:
            선택된 플랜 Dict (Planner의 출력 형식과 동일)
        """
        # ContextVar는 스레드 간 전달되지 않으므로, asyncio.to_thread()로 실행될 때를 대비해 명시적으로 설정
        set_session_id(session_id or "default")

        if not plans:
            raise ValueError("No plans to evaluate")

        if len(plans) == 1:
            # 플랜이 1개면 바로 반환
            return plans[0]

        # System Prompt 구성 (사용자 정보 포함)
        system_prompt = self.get_system_prompt(conversation_history, session_id)

        # 입력 메시지 구성 (role 기반 메시지 배열 - 가이드 권장 방식)
        # 요약본 + 최근 메시지 조합 사용
        effective_history = self.get_effective_history(
            conversation_history, session_id, recent_count=5
        )
        input_messages = []
        if effective_history:
            input_messages.extend(effective_history)

        # 평가용 메시지 추가
        # 성능 최적화: 평가에 필요한 필드만 추출, dialogue 길이 제한으로 토큰 수 감소
        evaluation_plans = []
        rag_context = None
        for plan in plans:
            if rag_context is None:
                rag_context = plan.get("rag_context")
            eval_plan = {
                "plan_id": plan.get("plan_id"),
                "intent": plan.get("intent"),
                "strategic_approach": plan.get("strategic_approach"),
                "spin_stage": plan.get("spin_stage"),
                "nudge_technique": plan.get("nudge_technique"),
                "dialogue": (plan.get("dialogue") or ""),
                "tool_calls": plan.get("tool_calls", []),
                "why_this_strategy": (plan.get("why_this_strategy") or ""),
            }
            evaluation_plans.append(eval_plan)

        rag_block = rag_context or "(RAG 검색 결과 없음)"
        evaluation_content = f"""
[사용자 메시지]
{user_message}

[RAG 검색 결과 — faithfulness/factuality 대조 근거]
{rag_block}

[평가할 플랜들]
Planner가 생성한 플랜들을 평가하세요.
각 플랜은 plan_id ("A", "B", "C")로 식별됩니다.

{json.dumps(evaluation_plans, ensure_ascii=False, indent=2)}

위 플랜들을 데이터 무결성(faithfulness, factuality) / 인텐트 정합성 / 개인화 설득 효용성으로 평가한 뒤,
가장 적합한 1개의 플랜을 선택하세요.
factuality가 1점(거짓)인 플랜은 총점 0으로 처리하세요.
선택한 플랜의 plan_id를 EvaluatorResponse.selected_plan에 설정하세요.
"""
        input_messages.append({"role": "user", "content": evaluation_content})

        try:
            evaluator_response = self.llm_client.generate_structured(
                messages=input_messages,
                response_schema=EvaluatorResponse,
                system_instruction=system_prompt,
                use_fast_model=True,
                session_id=session_id,
            )

            selected_plan_id = evaluator_response.selected_plan

            def _integrity_ok(evaluation) -> bool:
                try:
                    factuality = evaluation.scores.데이터_무결성.factuality
                    if factuality <= 1:
                        return False
                    if evaluation.scores.총점 == 0:
                        return False
                except Exception:
                    return True
                return True

            selected_evaluation = next(
                (
                    e
                    for e in evaluator_response.evaluations
                    if e.plan_id == selected_plan_id
                ),
                (
                    evaluator_response.evaluations[0]
                    if evaluator_response.evaluations
                    else None
                ),
            )
            if selected_evaluation and not _integrity_ok(selected_evaluation):
                fallback_eval = next(
                    (
                        e
                        for e in sorted(
                            evaluator_response.evaluations,
                            key=lambda x: x.scores.총점,
                            reverse=True,
                        )
                        if _integrity_ok(e)
                    ),
                    None,
                )
                if fallback_eval:
                    logger.warning(
                        "[Evaluator] %s는 factuality 실패, %s로 교체",
                        selected_plan_id,
                        fallback_eval.plan_id,
                    )
                    selected_plan_id = fallback_eval.plan_id
                    selected_evaluation = fallback_eval
            original_plan = next(
                (p for p in plans if p.get("plan_id") == selected_plan_id),
                plans[0],
            )

            # final_output을 Plan 형식으로 변환
            final_output = evaluator_response.final_output
            selected_plan_dict = original_plan.copy()

            # final_output의 dialogue를 사용 (더 적합한 경우)
            if final_output.dialogue:
                selected_plan_dict["dialogue"] = final_output.dialogue

            # spin_stage와 nudge_technique 업데이트
            if final_output.spin_stage:
                selected_plan_dict["spin_stage"] = final_output.spin_stage
            if final_output.nudge_technique:
                selected_plan_dict["nudge_technique"] = final_output.nudge_technique

            if selected_evaluation:
                logger.info(
                    f"[Evaluator] 선택된 플랜: {selected_plan_id}, "
                    f"총점: {selected_evaluation.scores.총점}, "
                    f"이유: {evaluator_response.selection_reasoning}"
                )
                logger.info(
                    f"[Evaluator] 선택된 플랜 상세 - dialogue: {selected_plan_dict.get('dialogue', 'NOT FOUND')}, "
                    f"spin_stage: {selected_plan_dict.get('spin_stage', 'NOT FOUND')}, "
                    f"nudge_technique: {selected_plan_dict.get('nudge_technique', 'NOT FOUND')}"
                )

            return selected_plan_dict

        except Exception as e:
            logger.error(f"[Evaluator] 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 첫 번째 플랜 반환
            return plans[0]
