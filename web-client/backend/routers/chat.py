"""
Chat API router with 3-stage Agent pipeline (Planner -> Evaluator -> Executor)

추가 요구사항:
- 세션 로그(JSON) export를 위해 assistant turn 기준으로
  - conversation: [{turn_id, speaker, text, timestamp}]
  - evidence: assistant_turn_id -> [{source_id, snippet_text}]
  - latency_ms: assistant_turn_id -> latency_ms
  - token_usage: assistant_turn_id -> {input, output, total}
를 기록한다.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import sys
from pathlib import Path
import os
import json
import asyncio
import base64
import time
from datetime import datetime, timezone
from typing import List, Optional
from openai import OpenAI
from openai import AsyncOpenAI

# 부모 디렉토리 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import (
    ChatRequest,
    ChatResponse,
    TTSStreamRequest,
    BatchExportRequest,
    BatchSaveRequest,
)
from services.llm_planner import LLMPlanner
from services.llm_evaluator import LLMEvaluator
from services.llm_executor import LLMExecutor
from services.llm.llm_usage_tracker import LLMUsageTracker
from utils.logging_context import set_session_id
from utils.session_log import SessionLogStore, ConversationTurn, EvidenceItem
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: Request, request: ChatRequest):
    """
    Chat endpoint with 3-stage Agent pipeline:
    1. Planner Agent: 최대 3개 시나리오 생성
    2. Evaluator Agent: 최종 1개 선택
    3. Executor Agent: 실행 및 결과 반환
    + 세션 로그 기록 (assistant turn 기준)
    """
    start_time = time.perf_counter()
    try:
        # 세션 ID는 SessionMiddleware가 쿠키에서 자동으로 읽어서 req.state.session_id에 설정
        # 서버 중심 방식: 쿠키(HttpOnly)로 세션 ID 관리 (단일 소스 진실)
        session_id = getattr(req.state, "session_id", None) or "default"
        set_session_id(session_id)  # 로깅 컨텍스트에 세션 ID 설정

        # Note: SessionMiddleware가 이미 세션 ID로 Redis를 저장/갱신했으므로
        # 여기서는 추가 Redis 작업 불필요 (중복 방지)

        # 메타데이터 필드 설정 (participant_id, user_goal_segment)
        if request.participant_id:
            SessionLogStore.set_participant_id(session_id, request.participant_id)
        if request.user_goal_segment:
            SessionLogStore.set_user_goal_segment(session_id, request.user_goal_segment)

        # 현재 assistant turn에 사용할 turn_id 생성
        assistant_turn_id = SessionLogStore.get_next_assistant_turn_id(session_id)

        # Step 1: Planner Agent - 최대 3개 시나리오 생성
        planner = LLMPlanner()
        plans = planner.plan(
            message=request.message,
            conversation_history=request.conversation_history,
            vehicle_id=request.vehicle_id,
            session_id=session_id,
        )

        selected_plan_id = None
        selection_reasoning = None

        if len(plans) == 1:
            # 시나리오가 1개면 바로 실행
            selected_plan = plans[0]
            selected_plan_id = selected_plan.get("plan_id", "A")
        else:
            # Step 2: Evaluator Agent - 최종 1개 선택
            evaluator = LLMEvaluator()
            selected_plan = evaluator.evaluate(
                plans=plans,
                user_message=request.message,
                conversation_history=request.conversation_history,
                session_id=session_id,
            )
            # selected_plan에서 plan_id 추출
            selected_plan_id = selected_plan.get(
                "plan_id", plans[0].get("plan_id", "A") if plans else "A"
            )

        # Step 3: Executor Agent - 실행
        executor = LLMExecutor()
        result = executor.execute(
            selected_plan=selected_plan,
            conversation_history=request.conversation_history,
            vehicle_id=request.vehicle_id,
            session_id=session_id,
        )

        # latency 측정 (엔드투엔드 기준)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 토큰 사용량: 이번 요청 동안 누적된 세션 usage를 assistant_turn_id에 매핑
        token_usage = LLMUsageTracker.pop_session_usage(session_id)

        # 세션 로그에 conversation / evidence / latency / token_usage 기록
        now_iso = datetime.now(timezone.utc).isoformat()

        # 사용자 턴
        SessionLogStore.add_turn(
            session_id,
            ConversationTurn(
                turn_id=f"user-{assistant_turn_id}",
                speaker="user",
                text=request.message,
                timestamp=now_iso,
            ),
        )

        # assistant 턴
        response_text = result.get("response", "")
        # selected_plan에서 lead_status 가져오기 (assistant 턴에 사용자 리드 상태 포함)
        lead_status = selected_plan.get("lead_status") if selected_plan else None
        SessionLogStore.add_turn(
            session_id,
            ConversationTurn(
                turn_id=assistant_turn_id,
                speaker="assistant",
                text=response_text,
                timestamp=now_iso,
                lead_status=lead_status,
            ),
        )

        # evidence 매핑 (assistant_turn_id -> [EvidenceItem])
        # evidence는 assistant 답변 turn에만 매핑됨
        evidence_items = []
        for ev in result.get("evidence", []):
            # source_id 추출: 우선순위 source_id > doc_id > vin > id
            source_id = str(
                ev.get("source_id")
                or ev.get("doc_id")
                or ev.get("vin")
                or ev.get("id")
                or ""
            )
            # snippet_text 추출: 원문 근거 스니펫 (우선순위 snippet_text > snippet > content > text)
            snippet_text = (
                ev.get("snippet_text")
                or ev.get("snippet")
                or ev.get("content")
                or ev.get("text")
                or ""
            )
            evidence_items.append(
                EvidenceItem(source_id=source_id, snippet_text=snippet_text)
            )
        if evidence_items:
            SessionLogStore.set_evidence(
                session_id=session_id,
                assistant_turn_id=assistant_turn_id,
                items=evidence_items,
            )

        # latency / token_usage 기록
        SessionLogStore.set_latency(
            session_id=session_id,
            assistant_turn_id=assistant_turn_id,
            latency_ms=latency_ms,
        )
        SessionLogStore.set_token_usage(
            session_id=session_id,
            assistant_turn_id=assistant_turn_id,
            usage=token_usage,
        )

        # planning_info 기록 (모든 생성된 plans와 선택된 plan 정보)
        SessionLogStore.set_planning_info(
            session_id=session_id,
            assistant_turn_id=assistant_turn_id,
            all_plans=plans,
            selected_plan_id=selected_plan_id or "A",
            selection_reasoning=selection_reasoning,
        )

        # Step 4: Tool calls 처리
        # searchVehicleDatabase는 Executor에서 이미 실행되었으므로 제외
        # 나머지 tool_calls는 프론트엔드로 전달
        executed_tool_calls = []
        for tool_call in result.get("tool_calls", []):
            name = tool_call.get("name")

            # searchVehicleDatabase는 Executor에서 이미 실행되었으므로 제외
            if name == "searchVehicleDatabase":
                continue

            # 나머지 tool들은 프론트엔드로 전달
            executed_tool_calls.append(
                {
                    "name": name,
                    "arguments": tool_call.get("arguments", {}),
                    "result": None,  # 프론트엔드에서 처리
                }
            )

        return ChatResponse(
            response=response_text,
            tool_calls=executed_tool_calls,
            evidence=result.get("evidence", []),
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream_endpoint(req: Request, request: ChatRequest):
    """
    Streaming chat endpoint with progressive dialogue updates
    Uses Server-Sent Events (SSE) for real-time streaming

    Streams dialogue word-by-word for better perceived latency
    """

    async def generate():
        """Generate chat response and stream dialogue progressively with heartbeat to prevent timeout"""
        start_time = time.perf_counter()
        try:
            # 세션 ID는 SessionMiddleware가 쿠키에서 자동으로 읽어서 req.state.session_id에 설정
            # 서버 중심 방식: 쿠키(HttpOnly)로 세션 ID 관리 (단일 소스 진실)
            session_id = getattr(req.state, "session_id", None) or "default"
            set_session_id(session_id)  # 로깅 컨텍스트에 세션 ID 설정

            # Note: SessionMiddleware가 이미 세션 ID로 Redis를 저장/갱신했으므로
            # 여기서는 추가 Redis 작업 불필요 (중복 방지)

            # 메타데이터 필드 설정 (participant_id, user_goal_segment)
            if request.participant_id:
                SessionLogStore.set_participant_id(session_id, request.participant_id)
            if request.user_goal_segment:
                SessionLogStore.set_user_goal_segment(
                    session_id, request.user_goal_segment
                )

            # 현재 assistant turn에 사용할 turn_id 생성
            assistant_turn_id = SessionLogStore.get_next_assistant_turn_id(session_id)

            # Heartbeat 설정: Vercel 10초 timeout 방지를 위해 4초마다 heartbeat 전송
            heartbeat_interval = 4.0  # 4초마다 heartbeat (10초 timeout 안전 마진)

            # Stage 1: Planner - 진행 상황 스트리밍
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'planner', 'status': 'started'}, ensure_ascii=False)}\n\n"

            # Planner 작업 시작 (비동기 태스크로 실행)
            planner = LLMPlanner()
            planner_task = asyncio.create_task(
                asyncio.to_thread(
                    planner.plan,
                    message=request.message,
                    conversation_history=request.conversation_history,
                    vehicle_id=request.vehicle_id,
                    session_id=session_id,
                )
            )

            # Planner 작업 중 heartbeat 전송 (Vercel timeout 방지)
            last_heartbeat_time = asyncio.get_event_loop().time()
            while not planner_task.done():
                await asyncio.sleep(0.5)  # 0.5초마다 체크 (더 빠른 반응)

                # 4초마다 heartbeat 전송 (SSE comment 형태: 브라우저에서 무시되지만 연결 유지)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    yield ": heartbeat\n\n"  # SSE heartbeat (Vercel timeout 방지)
                    last_heartbeat_time = current_time

            # Planner 결과 기다림
            plans = await planner_task

            yield f"data: {json.dumps({'type': 'stage', 'stage': 'planner', 'status': 'completed', 'plans_count': len(plans)}, ensure_ascii=False)}\n\n"

            # Stage 2: Evaluator (필요한 경우만)
            selected_plan_id = None

            if len(plans) == 1:
                # 플랜이 1개면 바로 실행
                selected_plan = plans[0]
                selected_plan_id = selected_plan.get("plan_id", "A")
                yield f"data: {json.dumps({'type': 'stage', 'stage': 'evaluator', 'status': 'skipped', 'reason': 'single_plan'}, ensure_ascii=False)}\n\n"
            else:
                # Evaluator 실행
                yield f"data: {json.dumps({'type': 'stage', 'stage': 'evaluator', 'status': 'started'}, ensure_ascii=False)}\n\n"

                evaluator = LLMEvaluator()
                evaluator_task = asyncio.create_task(
                    asyncio.to_thread(
                        evaluator.evaluate,
                        plans=plans,
                        user_message=request.message,
                        conversation_history=request.conversation_history,
                        session_id=session_id,
                    )
                )

                # Evaluator 작업 중 heartbeat 전송 (Vercel timeout 방지)
                last_heartbeat_time = asyncio.get_event_loop().time()
                while not evaluator_task.done():
                    await asyncio.sleep(0.5)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        yield ": heartbeat\n\n"
                        last_heartbeat_time = current_time

                selected_plan = await evaluator_task
                # selected_plan에서 plan_id 추출
                selected_plan_id = selected_plan.get(
                    "plan_id", plans[0].get("plan_id", "A") if plans else "A"
                )

                yield f"data: {json.dumps({'type': 'stage', 'stage': 'evaluator', 'status': 'completed'}, ensure_ascii=False)}\n\n"

            # Stage 3: Executor - 실행
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'executor', 'status': 'started'}, ensure_ascii=False)}\n\n"

            executor = LLMExecutor()
            executor_task = asyncio.create_task(
                asyncio.to_thread(
                    executor.execute,
                    selected_plan=selected_plan,
                    conversation_history=request.conversation_history,
                    vehicle_id=request.vehicle_id,
                    session_id=session_id,
                )
            )

            # Executor 작업 중 heartbeat 전송 (Vercel timeout 방지)
            last_heartbeat_time = asyncio.get_event_loop().time()
            while not executor_task.done():
                await asyncio.sleep(0.5)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat_time = current_time

            result = await executor_task

            # latency 측정 (엔드투엔드 기준: planner ~ executor)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # 토큰 사용량: 이번 요청 동안 누적된 세션 usage를 assistant_turn_id에 매핑
            token_usage = LLMUsageTracker.pop_session_usage(session_id)

            # Stage 4: Dialogue 스트리밍
            dialogue = result.get("response", "")

            if dialogue:
                # Dialogue를 문자 단위로 스트리밍 (한국어/영어 모두 자연스럽게)
                # 한국어는 공백으로 단어를 구분하지 않으므로 문자 단위가 더 자연스러움
                accumulated_text = ""
                chunk_size = 3  # 한 번에 3글자씩 전송 (더 빠른 체감)

                # Dialogue 스트리밍 중에도 heartbeat 체크 (긴 dialogue의 경우)
                dialogue_start_time = asyncio.get_event_loop().time()
                last_heartbeat_time = dialogue_start_time

                for i in range(0, len(dialogue), chunk_size):
                    chunk = dialogue[i : i + chunk_size]
                    accumulated_text += chunk

                    # 문자 단위로 스트리밍
                    yield f"data: {json.dumps({'type': 'dialogue_chunk', 'text': chunk, 'accumulated': accumulated_text}, ensure_ascii=False)}\n\n"

                    # 자연스러운 타이핑 속도 (3글자당 약 20-30ms)
                    await asyncio.sleep(0.02)

                    # Dialogue 스트리밍 중에도 heartbeat 전송 (긴 텍스트의 경우 timeout 방지)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        yield ": heartbeat\n\n"
                        last_heartbeat_time = current_time

            # Stage 5: Tool calls 및 Evidence 전송
            executed_tool_calls = []
            for tool_call in result.get("tool_calls", []):
                name = tool_call.get("name")

                # searchVehicleDatabase는 Executor에서 이미 실행되었으므로 제외
                if name == "searchVehicleDatabase":
                    continue

                executed_tool_calls.append(
                    {
                        "name": name,
                        "arguments": tool_call.get("arguments", {}),
                        "result": None,
                    }
                )

            # Tool calls 전송
            if executed_tool_calls:
                yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': executed_tool_calls}, ensure_ascii=False)}\n\n"

            # Evidence 전송
            evidence = result.get("evidence", [])
            if evidence:
                yield f"data: {json.dumps({'type': 'evidence', 'evidence': evidence}, ensure_ascii=False)}\n\n"

            # 세션 로그에 conversation / evidence / latency / token_usage 기록
            now_iso = datetime.now(timezone.utc).isoformat()

            # 사용자 턴
            SessionLogStore.add_turn(
                session_id,
                ConversationTurn(
                    turn_id=f"user-{assistant_turn_id}",
                    speaker="user",
                    text=request.message,
                    timestamp=now_iso,
                ),
            )

            # assistant 턴
            # selected_plan에서 lead_status 가져오기 (assistant 턴에 사용자 리드 상태 포함)
            lead_status = selected_plan.get("lead_status") if selected_plan else None
            SessionLogStore.add_turn(
                session_id,
                ConversationTurn(
                    turn_id=assistant_turn_id,
                    speaker="assistant",
                    text=dialogue or "",
                    timestamp=now_iso,
                    lead_status=lead_status,
                ),
            )

            # evidence 매핑 (assistant 답변 turn에만 매핑)
            evidence_items = []
            for ev in evidence or []:
                # source_id 추출: 우선순위 source_id > doc_id > vin > id
                source_id = str(
                    ev.get("source_id")
                    or ev.get("doc_id")
                    or ev.get("vin")
                    or ev.get("id")
                    or ""
                )
                # snippet_text 추출: 원문 근거 스니펫 (우선순위 snippet_text > snippet > content > text)
                snippet_text = (
                    ev.get("snippet_text")
                    or ev.get("snippet")
                    or ev.get("content")
                    or ev.get("text")
                    or ""
                )
                evidence_items.append(
                    EvidenceItem(source_id=source_id, snippet_text=snippet_text)
                )
            if evidence_items:
                SessionLogStore.set_evidence(
                    session_id=session_id,
                    assistant_turn_id=assistant_turn_id,
                    items=evidence_items,
                )

            # latency / token_usage 기록
            SessionLogStore.set_latency(
                session_id=session_id,
                assistant_turn_id=assistant_turn_id,
                latency_ms=latency_ms,
            )
            SessionLogStore.set_token_usage(
                session_id=session_id,
                assistant_turn_id=assistant_turn_id,
                usage=token_usage,
            )

            # planning_info 기록 (모든 생성된 plans와 선택된 plan 정보)
            SessionLogStore.set_planning_info(
                session_id=session_id,
                assistant_turn_id=assistant_turn_id,
                all_plans=plans,
                selected_plan_id=selected_plan_id or "A",
                selection_reasoning=None,  # evaluator.evaluate()는 selection_reasoning을 반환하지 않음
            )

            # 완료 신호
            yield f"data: {json.dumps({'type': 'complete'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            # 에러 전송
            error_data = {"type": "error", "message": str(e), "stage": "unknown"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/session-log/{session_id}")
async def get_session_log(session_id: str):
    """
    세션 로그 JSON export 엔드포인트.

    반환 스키마 예시:
    {
      "session_id": "...",
      "conversation": [{turn_id, speaker, text, timestamp}, ...],
      "evidence": {assistant_turn_id: [{source_id, snippet_text}, ...]},
      "latency_ms": {assistant_turn_id: 123.4},
      "token_usage": {assistant_turn_id: {"input": x, "output": y, "total": z}}
    }

    Note: 세션 로그는 첫 번째 채팅 요청이 들어올 때 생성됩니다.
    세션이 없으면 빈 세션을 생성하여 반환합니다.
    """
    data = SessionLogStore.to_dict(session_id)
    if not data:
        # 세션이 없으면 빈 세션 생성 (조회 목적)
        SessionLogStore.get_or_create(session_id)
        data = SessionLogStore.to_dict(session_id)
        if not data:
            available_sessions = SessionLogStore.list_all_sessions()
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found and could not be created. "
                f"Available sessions: {available_sessions[:10] if available_sessions else 'none'}. "
                f"Total sessions: {len(available_sessions)}",
            )
    return data


@router.get("/session-log")
async def list_all_sessions():
    """
    현재 메모리에 있는 모든 세션 ID 리스트 반환.
    (디버깅용)

    Note: 세션 로그는 첫 번째 채팅 요청이 들어올 때 생성됩니다.
    프론트엔드에서 세션 ID를 생성했더라도, 실제 채팅 요청이 없으면
    세션 로그가 메모리에 존재하지 않습니다.
    """
    sessions = SessionLogStore.list_all_sessions()
    return {
        "total": len(sessions),
        "sessions": sessions,
        "note": "Sessions are created when the first chat request is made. "
        "A session ID created in the frontend won't appear here until a chat message is sent.",
    }


@router.post("/session-log/batch-export")
async def batch_export_session_logs(request: BatchExportRequest):
    """
    여러 세션을 한 번에 export하는 배치 엔드포인트 (메모리에서 읽기만, 파일 저장 없음).

    Request Body (JSON):
    {
      "session_ids": ["session_1", "session_2", ...]  # None이면 모든 세션
    }

    Returns:
    {
      "session_1": {session_log_data},
      "session_2": {session_log_data},
      ...
    }
    """
    result = SessionLogStore.batch_export(request.session_ids)
    return result


@router.post("/session-log/batch-save")
async def batch_save_session_logs(request: BatchSaveRequest):
    """
    여러 세션을 한 번에 파일로 저장하는 배치 엔드포인트.

    Request Body (JSON):
    {
      "session_ids": ["session_1", "session_2", ...],  # None이면 모든 세션
      "delete_after_save": false  # 저장 후 메모리에서 삭제할지 여부 (기본값: false)
    }

    Returns:
    {
      "saved_count": 3,
      "failed_count": 1,
      "saved_sessions": [
        {"session_id": "session_1", "filepath": "logs/session_logs/session_log_xxx.json"},
        ...
      ],
      "failed_sessions": ["session_2", ...]
    }
    """
    result = SessionLogStore.batch_save_to_file(
        session_ids=request.session_ids,
        delete_after_save=request.delete_after_save,
    )
    return result


@router.post("/session-log/{session_id}/save")
async def save_session_log_to_file(session_id: str):
    """
    세션 로그를 JSON 파일로 저장하고 메모리에서 삭제.
    (명시적 세션 종료 시 사용)

    Returns:
    {
      "success": true,
      "filepath": "logs/session_logs/session_log_xxx_1234567890.json"
    }
    """
    # 먼저 세션이 존재하는지 확인
    available_sessions = SessionLogStore.list_all_sessions()
    if session_id not in available_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found in memory. "
            f"Available sessions: {available_sessions[:10] if available_sessions else 'none'}. "
            f"Total sessions: {len(available_sessions)}",
        )

    filepath = SessionLogStore.force_save_and_delete(session_id)
    if not filepath:
        # 파일 저장 실패 - 서버 로그 확인 필요
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"Failed to save session {session_id} to file. "
            f"LOG_DIR: {SessionLogStore.LOG_DIR}, "
            f"LOG_DIR exists: {SessionLogStore.LOG_DIR.exists()}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save session log to file. "
            f"LOG_DIR: {SessionLogStore.LOG_DIR}, "
            f"Check server logs for details.",
        )
    return {"success": True, "filepath": filepath}


@router.post("/session-log/cleanup")
async def cleanup_expired_sessions():
    """
    만료된 세션(TTL 초과)을 파일로 저장하고 메모리에서 삭제.
    (주기적으로 호출하거나 수동으로 호출)

    Returns:
    {
      "saved_count": 3,
      "saved_sessions": ["session_1", "session_2", "session_3"]
    }
    """
    saved_sessions = SessionLogStore.cleanup_and_save_expired_sessions()
    return {"saved_count": len(saved_sessions), "saved_sessions": saved_sessions}


@router.post("/tts/stream-chunks")
async def stream_tts_chunks(request: TTSStreamRequest):
    """
    Stream TTS audio chunks for presentation script segments
    Uses Server-Sent Events (SSE) for real-time streaming

    Called by frontend PresentationPlayer after receiving generatePresentationScript tool call
    """
    # ===== 🔍 TTS Request Dictionary 로깅 (전체 출력, 스킵 없음) =====
    logger.info("=" * 80)
    logger.info("[TTS] 📥 Received generatePresentationScript Dictionary:")
    logger.info(f"  - Total Segments: {len(request.segments)}")
    logger.info(f"  - Voice: {request.voice}")
    logger.info(f"  - Model: {request.model}")
    logger.info(f"  - Speed: {request.speed}")
    logger.info(f"  - Instructions: {request.instructions}")
    logger.info("")

    # 모든 Segments 상세 정보 출력 (스킵 없음)
    for i, segment in enumerate(request.segments):
        logger.info(f"  [Segment {i}]")
        logger.info(f"    - Timestamp: {segment.timestamp}")
        logger.info(f"    - Text: {segment.text}")  # 전체 텍스트 출력
        logger.info(f"    - Actions count: {len(segment.actions)}")

        # 모든 Actions 상세 출력
        for j, action in enumerate(segment.actions):
            # Pydantic 모델을 dict로 변환
            action_dict = (
                action.model_dump() if hasattr(action, "model_dump") else action
            )
            logger.info(f"      [Action {j}]: {action_dict}")
        logger.info("")  # 세그먼트 간 구분을 위한 빈 줄

    logger.info("=" * 80)
    # ===== End of Logging =====

    # Initialize OpenAI client (async for streaming)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    client = AsyncOpenAI(api_key=api_key)

    async def generate():
        """Generate TTS audio chunks and stream them via SSE using real-time streaming"""
        try:
            for i, segment in enumerate(request.segments):
                text = segment.text
                if not text:
                    continue

                # Generate TTS for this chunk using streaming response
                try:
                    # Prepare parameters for TTS request
                    tts_params = {
                        "model": request.model or "gpt-4o-mini-tts",
                        "voice": request.voice or "alloy",
                        "input": text,
                        "speed": request.speed or 1.0,
                        "response_format": "wav",  # Use WAV for low latency (no decoding overhead)
                    }

                    # Only include instructions if it's provided and not empty
                    if request.instructions and request.instructions.strip():
                        tts_params["instructions"] = request.instructions.strip()

                    # Use with_streaming_response for real-time audio streaming
                    # This allows audio to be played before the full file is generated
                    # Set timeout to prevent hanging (30 seconds per segment)
                    async with client.audio.speech.with_streaming_response.create(
                        **tts_params, timeout=30.0  # 30초 타임아웃 (세그먼트당)
                    ) as response:
                        # Collect all audio chunks for this segment
                        audio_chunks = []
                        async for chunk in response.iter_bytes():
                            if chunk:
                                audio_chunks.append(chunk)

                        # Combine all chunks into single audio data
                        audio_data = b"".join(audio_chunks)

                        # Convert to base64 for JSON transmission
                        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                        # Send chunk with metadata
                        # Handle actions safely (may be None or empty list)
                        actions_list = []
                        if segment.actions:
                            for action in segment.actions:
                                if hasattr(action, "model_dump"):
                                    actions_list.append(action.model_dump())
                                elif hasattr(action, "dict"):
                                    actions_list.append(action.dict())
                                elif isinstance(action, dict):
                                    actions_list.append(action)
                                else:
                                    # Fallback: try to convert to dict
                                    actions_list.append(action)

                        chunk_data = {
                            "type": "chunk",
                            "index": i,
                            "text": text,
                            "timestamp": segment.timestamp,
                            "audio_base64": audio_base64,
                            "actions": actions_list,
                        }

                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

                        # Small delay between chunks to prevent overwhelming
                        await asyncio.sleep(0.1)

                except Exception as e:
                    # Send error for this chunk but continue
                    error_data = {
                        "type": "error",
                        "index": i,
                        "message": str(e),
                        "text": text,
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    continue

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            # Send final error
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
