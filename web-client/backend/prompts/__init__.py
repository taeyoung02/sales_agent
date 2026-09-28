"""
프롬프트 모듈
모든 프롬프트를 중앙에서 관리하여 외부 업데이트 시 자동 적용되도록 구성
"""

# 공통 프롬프트
from .base import PROMPT_REGISTRY
from .tools import TOOL_REGISTRY

# Full 버전 프롬프트만 사용
from .full import (
    get_sales_knowledge_full as get_sales_knowledge,
    get_common_tool_guidelines_full as get_common_tool_guidelines,
    get_planner_tool_guidelines_full as get_planner_tool_guidelines,
    get_planner_role_full as get_planner_role,
    get_lead_classification_prompt_full as get_lead_classification_prompt,
    get_evaluator_role,
)

__all__ = [
    "get_sales_knowledge",
    "get_common_tool_guidelines",
    "get_planner_tool_guidelines",
    "get_planner_role",
    "get_lead_classification_prompt",
    "get_evaluator_role",
    "PROMPT_REGISTRY",
    "TOOL_REGISTRY",
]
