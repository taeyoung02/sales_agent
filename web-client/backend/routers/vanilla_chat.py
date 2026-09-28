"""
Vanilla Chat API router - Simple RAG-only chatbot
단순 RAG 검색만 수행하여 답변을 생성하는 챗봇
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import sys
from pathlib import Path
import os
import json
import asyncio
import time
from datetime import datetime, timezone
from typing import List, Optional
from openai import AsyncOpenAI

# 부모 디렉토리 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import ChatRequest
from rag.rag import get_rag_pipeline
from utils.logging_context import set_session_id
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vanilla-chat", tags=["vanilla-chat"])

# Vanilla chatbot system prompt
VANILLA_SYSTEM_PROMPT = """당신은 중고차 정보를 제공하는 친절한 챗봇입니다.
사용자의 질문에 대해 제공된 컨텍스트 정보를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

답변 시 다음 사항을 지키세요:
- 제공된 컨텍스트에서 찾은 정보를 정확하게 전달하세요
- 컨텍스트에 없는 정보는 추측하지 말고, "제공된 정보에서는 해당 내용을 찾을 수 없습니다"라고 명확히 말하세요
- 친절하고 자연스러운 대화체로 답변하세요
- 사용자가 요청한 차량 정보가 있으면 명확하게 정리해서 보여주세요"""


@router.post("/stream")
async def vanilla_chat_stream_endpoint(req: Request, request: ChatRequest):
    """
    Vanilla chat streaming endpoint - Simple RAG-only chatbot
    RAG 검색만 수행하고 LLM으로 답변을 생성하여 스트리밍으로 반환
    """

    async def generate():
        """Generate vanilla chat response with RAG search"""
        start_time = time.perf_counter()
        try:
            # 세션 ID는 SessionMiddleware가 쿠키에서 자동으로 읽어서 req.state.session_id에 설정
            session_id = getattr(req.state, "session_id", None) or "default"
            set_session_id(session_id)

            # Stage 1: RAG 검색
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'rag_search', 'status': 'started'})}\n\n"

            rag_pipeline = get_rag_pipeline()
            rag_context = rag_pipeline.get_context(
                query=request.message,
                n_results=3,
            )

            logger.info(
                f"[VanillaChat] RAG search completed. Context length: {len(rag_context)}"
            )
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'rag_search', 'status': 'completed'})}\n\n"

            # Evidence formatting
            evidence = []
            if rag_context and rag_context != "No relevant context found.":
                # RAG 검색 결과에서 evidence 정보 추출 (간단한 형식)
                evidence.append(
                    {
                        "title": "검색 결과",
                        "snippet": (
                            rag_context[:500] if len(rag_context) > 500 else rag_context
                        ),
                        "asOf": "Current",
                    }
                )

            # Evidence 전송
            if evidence:
                yield f"data: {json.dumps({'type': 'evidence', 'evidence': evidence})}\n\n"

            # Stage 2: 답변 생성
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'response_generation', 'status': 'started'})}\n\n"

            # System prompt에 RAG context 포함
            system_prompt = (
                f"{VANILLA_SYSTEM_PROMPT}\n\n=== 컨텍스트 정보 ===\n{rag_context}"
            )

            # 대화 히스토리 구성
            input_messages = []
            if request.conversation_history:
                # 최근 10개 메시지만 사용 (토큰 절약)
                recent_history = request.conversation_history[-10:]
                input_messages.extend(recent_history)
            else:
                input_messages = []

            # 현재 사용자 메시지 추가
            input_messages.append({"role": "user", "content": request.message})

            # Streaming으로 답변 생성
            accumulated_text = ""
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            # AsyncOpenAI를 사용하여 스트리밍
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=500, detail="OPENAI_API_KEY not configured"
                )

            async_client = AsyncOpenAI(api_key=api_key)

            # OpenAI 스트리밍 API 사용
            stream = await async_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}]
                + input_messages,
                stream=True,
                temperature=0.7,
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        accumulated_text += delta.content
                        yield f"data: {json.dumps({'type': 'dialogue_chunk', 'text': delta.content, 'accumulated': accumulated_text})}\n\n"

            # Stage 2 완료
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'response_generation', 'status': 'completed'})}\n\n"

            # 완료
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Vanilla chat stream error: {e}", exc_info=True)
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("")
async def vanilla_chat_endpoint(req: Request, request: ChatRequest):
    """
    Vanilla chat endpoint (non-streaming) - Simple RAG-only chatbot
    """
    start_time = time.perf_counter()
    try:
        # 세션 ID는 SessionMiddleware가 쿠키에서 자동으로 읽어서 req.state.session_id에 설정
        session_id = getattr(req.state, "session_id", None) or "default"
        set_session_id(session_id)

        # RAG 검색
        rag_pipeline = get_rag_pipeline()
        rag_context = rag_pipeline.get_context(
            query=request.message,
            n_results=3,
        )

        logger.info(
            f"[VanillaChat] RAG search completed. Context length: {len(rag_context)}"
        )

        # Evidence formatting
        evidence = []
        if rag_context and rag_context != "No relevant context found.":
            evidence.append(
                {
                    "title": "검색 결과",
                    "snippet": (
                        rag_context[:500] if len(rag_context) > 500 else rag_context
                    ),
                    "asOf": "Current",
                }
            )

        # LLM으로 답변 생성
        from services.llm import LLMClientFactory

        llm_client = LLMClientFactory.create_client()

        system_prompt = (
            f"{VANILLA_SYSTEM_PROMPT}\n\n=== 컨텍스트 정보 ===\n{rag_context}"
        )

        input_messages = []
        if request.conversation_history:
            recent_history = request.conversation_history[-10:]
            input_messages.extend(recent_history)

        input_messages.append({"role": "user", "content": request.message})

        # 답변 생성
        response_text = llm_client.generate_text(
            messages=input_messages,
            system_instruction=system_prompt,
            use_fast_model=True,
            session_id=session_id,
        )

        return {
            "response": response_text,
            "tool_calls": [],  # Vanilla chatbot은 tool calls 없음
            "evidence": evidence,
        }

    except Exception as e:
        logger.error(f"Vanilla chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
