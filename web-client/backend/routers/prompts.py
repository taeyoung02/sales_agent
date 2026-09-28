"""
프롬프트 관리 API router
프롬프트 조회 및 업데이트 기능 제공
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import sys
from pathlib import Path
import importlib

# 부모 디렉토리 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# Full 버전 프롬프트만 사용

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


class PromptUpdateRequest(BaseModel):
    """프롬프트 업데이트 요청"""

    sales_knowledge: Optional[str] = None
    common_tool_guidelines: Optional[str] = None
    planner_tool_guidelines: Optional[str] = None
    planner_role: Optional[str] = None
    evaluator_role: Optional[str] = None


class PromptsResponse(BaseModel):
    """프롬프트 조회 응답"""

    sales_knowledge: str
    common_tool_guidelines: str
    planner_tool_guidelines: str
    planner_role: str
    evaluator_role: str


class PromptUpdateResponse(BaseModel):
    """프롬프트 업데이트 응답"""

    status: str
    message: str
    updates: Dict[str, str]


def _extract_prompt_from_file(file_content: str, function_name: str) -> str:
    """
    파일 내용에서 프롬프트 함수의 return 값 추출
    파일을 직접 파싱하므로 항상 최신 내용을 가져올 수 있음
    """
    import re

    # 함수 정의 패턴: def function_name(...) -> str:
    function_def_pattern = rf"def {function_name}\(.*?\) -> str:"

    # 함수 정의 위치 찾기
    def_match = re.search(function_def_pattern, file_content)
    if not def_match:
        raise ValueError(f"함수 {function_name}을 찾을 수 없습니다.")

    # 함수 본문 시작 위치 찾기 (docstring 포함)
    body_start = def_match.end()

    # 다음 함수 정의나 파일 끝까지 찾기
    next_function_pattern = r"\n\ndef \w+\(|$"
    next_match = re.search(next_function_pattern, file_content[body_start:])

    if next_match:
        function_end = body_start + next_match.start()
    else:
        function_end = len(file_content)

    # 함수 본문에서 return """...""" 패턴 찾기
    function_body = file_content[body_start:function_end]

    # return """ ... """ 패턴 (멀티라인, non-greedy)
    return_pattern = r'return\s+"""(.*?)"""'
    return_match = re.search(return_pattern, function_body, re.DOTALL)

    if not return_match:
        raise ValueError(f"함수 {function_name}의 return 문을 찾을 수 없습니다.")

    # triple quotes 내부의 내용 추출
    return return_match.group(1).strip()


@router.get("", response_model=PromptsResponse)
async def get_prompts():
    """
    모든 프롬프트 조회
    파일을 직접 읽어서 파싱하므로 항상 최신 내용을 반환
    """
    try:
        # Full 버전 프롬프트 파일 경로
        prompts_dir = Path(__file__).parent.parent / "prompts"
        sales_knowledge_file = prompts_dir / "full" / "sales_knowledge.py"
        tool_guidelines_file = prompts_dir / "full" / "tool_guidelines.py"
        planner_role_file = prompts_dir / "full" / "planner_role.py"
        evaluator_role_file = prompts_dir / "full" / "evaluator_role.py"

        # 각 파일에서 프롬프트 추출
        def read_prompt_from_file(file_path: Path, function_name: str) -> str:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"프롬프트 파일을 찾을 수 없습니다: {file_path}"
                )
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            return _extract_prompt_from_file(file_content, function_name)

        return PromptsResponse(
            sales_knowledge=read_prompt_from_file(
                sales_knowledge_file, "get_sales_knowledge_full"
            ),
            common_tool_guidelines=read_prompt_from_file(
                tool_guidelines_file, "get_common_tool_guidelines_full"
            ),
            planner_tool_guidelines=read_prompt_from_file(
                tool_guidelines_file, "get_planner_tool_guidelines_full"
            ),
            planner_role=read_prompt_from_file(
                planner_role_file, "get_planner_role_full"
            ),
            evaluator_role=read_prompt_from_file(
                evaluator_role_file, "get_evaluator_role"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프롬프트 조회 실패: {str(e)}")


@router.put("", response_model=PromptUpdateResponse)
async def update_prompts(request: PromptUpdateRequest):
    """
    프롬프트 업데이트
    업데이트되지 않은 필드는 현재 값 유지
    """
    try:
        # Full 버전 프롬프트 파일 경로
        prompts_dir = Path(__file__).parent.parent / "prompts"
        sales_knowledge_file = prompts_dir / "full" / "sales_knowledge.py"
        tool_guidelines_file = prompts_dir / "full" / "tool_guidelines.py"
        planner_role_file = prompts_dir / "full" / "planner_role.py"
        evaluator_role_file = prompts_dir / "full" / "evaluator_role.py"

        # 각 파일 읽기
        def read_file_content(file_path: Path) -> str:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"프롬프트 파일을 찾을 수 없습니다: {file_path}"
                )
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        sales_knowledge_content = read_file_content(sales_knowledge_file)
        tool_guidelines_content = read_file_content(tool_guidelines_file)
        planner_role_content = read_file_content(planner_role_file)
        evaluator_role_content = read_file_content(evaluator_role_file)

        # 각 프롬프트 함수 업데이트
        updates = {}

        if request.sales_knowledge is not None:
            sales_knowledge_content = _update_function(
                sales_knowledge_content,
                "get_sales_knowledge_full",
                request.sales_knowledge,
            )
            updates["sales_knowledge"] = "업데이트됨"
        else:
            updates["sales_knowledge"] = "유지됨"

        if request.common_tool_guidelines is not None:
            tool_guidelines_content = _update_function(
                tool_guidelines_content,
                "get_common_tool_guidelines_full",
                request.common_tool_guidelines,
            )
            updates["common_tool_guidelines"] = "업데이트됨"
        else:
            updates["common_tool_guidelines"] = "유지됨"

        if request.planner_tool_guidelines is not None:
            tool_guidelines_content = _update_function(
                tool_guidelines_content,
                "get_planner_tool_guidelines_full",
                request.planner_tool_guidelines,
            )
            updates["planner_tool_guidelines"] = "업데이트됨"
        else:
            updates["planner_tool_guidelines"] = "유지됨"

        if request.planner_role is not None:
            planner_role_content = _update_function(
                planner_role_content, "get_planner_role_full", request.planner_role
            )
            updates["planner_role"] = "업데이트됨"
        else:
            updates["planner_role"] = "유지됨"

        if request.evaluator_role is not None:
            evaluator_role_content = _update_function(
                evaluator_role_content, "get_evaluator_role", request.evaluator_role
            )
            updates["evaluator_role"] = "업데이트됨"
        else:
            updates["evaluator_role"] = "유지됨"

        # 파일 저장
        with open(sales_knowledge_file, "w", encoding="utf-8") as f:
            f.write(sales_knowledge_content)
        with open(tool_guidelines_file, "w", encoding="utf-8") as f:
            f.write(tool_guidelines_content)
        with open(planner_role_file, "w", encoding="utf-8") as f:
            f.write(planner_role_content)
        with open(evaluator_role_file, "w", encoding="utf-8") as f:
            f.write(evaluator_role_content)

        # 프롬프트 모듈 리로드
        import prompts

        importlib.reload(prompts)

        # Full 버전 서브모듈 리로드
        try:
            import prompts.full

            importlib.reload(prompts.full)
        except ImportError:
            pass

        return {
            "status": "success",
            "message": "프롬프트가 성공적으로 업데이트되었습니다.",
            "updates": updates,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프롬프트 업데이트 실패: {str(e)}")


def _update_function(file_content: str, function_name: str, new_content: str) -> str:
    """
    파일 내용에서 특정 함수의 반환값 업데이트
    """
    import re

    # 함수 정의 패턴: def function_name(...) -> str:
    function_def_pattern = rf"def {function_name}\(.*?\) -> str:"

    # 함수 정의 위치 찾기
    def_match = re.search(function_def_pattern, file_content)
    if not def_match:
        raise ValueError(f"함수 {function_name}을 찾을 수 없습니다.")

    function_start = def_match.start()

    # 함수 본문 시작 위치 찾기 (docstring 포함)
    body_start = def_match.end()

    # 다음 함수 정의나 파일 끝까지 찾기
    # \n\ndef 로 시작하는 다음 함수 찾기
    next_function_pattern = r"\n\ndef \w+\(|$"
    next_match = re.search(next_function_pattern, file_content[body_start:])

    if next_match:
        function_end = body_start + next_match.start()
    else:
        function_end = len(file_content)

    # 함수 본문에서 return """...""" 패턴 찾기
    function_body = file_content[body_start:function_end]

    # return """ ... """ 패턴 (멀티라인, non-greedy)
    # triple quotes 내부의 모든 내용을 매칭
    return_pattern = r'return\s+"""(.*?)"""'
    return_match = re.search(return_pattern, function_body, re.DOTALL)

    if not return_match:
        raise ValueError(f"함수 {function_name}의 return 문을 찾을 수 없습니다.")

    # return 문 교체
    # new_content에 이미 포함된 따옴표는 그대로 유지
    new_return = f'return """{new_content}"""'

    updated_function_body = (
        function_body[: return_match.start()]
        + new_return
        + function_body[return_match.end() :]
    )

    # 전체 파일 내용 업데이트
    return (
        file_content[:body_start] + updated_function_body + file_content[function_end:]
    )
