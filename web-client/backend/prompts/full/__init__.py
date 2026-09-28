"""
전체 버전 프롬프트 모듈
모든 상세 프롬프트를 포함하는 버전
"""

from .sales_knowledge import get_sales_knowledge_full
from .tool_guidelines import (
    get_common_tool_guidelines_full,
    get_planner_tool_guidelines_full,
)
from .planner_role import get_planner_role_full
from .planner_lead_classification import get_lead_classification_prompt_full
from .evaluator_role import get_evaluator_role

__all__ = [
    "get_sales_knowledge_full",
    "get_common_tool_guidelines_full",
    "get_planner_tool_guidelines_full",
    "get_planner_role_full",
    "get_lead_classification_prompt_full",
    "get_evaluator_role",
]
