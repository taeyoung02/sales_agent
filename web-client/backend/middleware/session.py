"""
FastAPI 세션 미들웨어

쿠키 기반 세션 ID 관리 및 Redis TTL 자동 갱신
"""

import uuid
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from utils.redis_client import RedisSessionManager

logger = logging.getLogger(__name__)


class SessionMiddleware(BaseHTTPMiddleware):
    """
    쿠키 기반 세션 ID 관리 미들웨어

    - 요청마다 세션 ID 확인 (쿠키 또는 헤더)
    - 세션 ID 없으면 UUID 생성
    - Redis에 세션 데이터 저장 및 TTL 갱신
    - 응답 쿠키에 session_id 설정 (HttpOnly)
    """

    # 세션 쿠키 이름
    SESSION_COOKIE_NAME = "session_id"

    # 세션 TTL (초) - RedisSessionManager와 동일
    SESSION_TTL = RedisSessionManager.SESSION_TTL

    # 세션 ID를 요구하지 않는 경로 (헬스체크 등)
    # 참고: /api/health는 제외하지 않음 - 세션 쿠키 설정은 필요하지만 Redis TTL 갱신은 생략
    EXCLUDED_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # Redis TTL 갱신을 생략하는 경로 (쿠키는 설정하되 Redis 작업만 생략)
    SKIP_REDIS_TTL_PATHS = {
        "/api/health",
    }

    async def dispatch(self, request: Request, call_next):
        """
        요청 처리 전후로 세션 관리
        """
        # 제외 경로는 세션 처리 완전히 스킵
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # 1. 세션 ID 가져오기 (쿠키 > 헤더 순서)
        session_id = request.cookies.get(self.SESSION_COOKIE_NAME)
        if not session_id:
            session_id = request.headers.get("X-Session-ID")

        # 2. 세션 ID 없으면 새로 생성
        if not session_id:
            session_id = str(uuid.uuid4())
        else:
            logger.debug(f"Existing session: {session_id}")

        # 3. Request state에 session_id 저장 (라우터에서 사용 가능)
        request.state.session_id = session_id

        # 4. Redis에 세션 데이터 확인 및 TTL 갱신 (SKIP_REDIS_TTL_PATHS 제외)
        if request.url.path not in self.SKIP_REDIS_TTL_PATHS:
            try:
                session_exists = await RedisSessionManager.session_exists(session_id)

                if session_exists:
                    # 기존 세션: TTL만 갱신
                    await RedisSessionManager.extend_session_ttl(session_id)
                    logger.debug(f"Session TTL extended: {session_id}")
                else:
                    # 새 세션: 초기 데이터 저장
                    initial_data = {
                        "created_at": (
                            request.state.request_timestamp
                            if hasattr(request.state, "request_timestamp")
                            else None
                        ),
                        "last_access": None,
                    }
                    await RedisSessionManager.set_session(session_id, initial_data)
                    logger.debug(f"New session initialized in Redis: {session_id}")
            except Exception as e:
                logger.warning(f"Redis unavailable for session {session_id}: {e}")
                # Redis 실패 시에도 요청은 계속 처리 (degraded mode)
                # 세션 ID는 쿠키에 저장되므로 메모리 기반 세션 관리는 가능
        else:
            # SKIP_REDIS_TTL_PATHS인 경우 Redis 작업 생략 (쿠키만 설정)
            logger.debug(f"Skipping Redis TTL update for {request.url.path}")

        # 5. 다음 미들웨어 또는 라우터 호출
        response: Response = await call_next(request)

        # 6. 응답 쿠키에 session_id 설정
        # HTTPS 환경 감지: 여러 헤더 확인 (Cloudflare, Nginx 등 다양한 프록시 지원)
        # Cloudflare는 X-Forwarded-Proto, CF-Visitor, 또는 X-Forwarded-Ssl 사용
        x_forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
        x_forwarded_ssl = request.headers.get("X-Forwarded-Ssl", "").lower()
        cf_visitor = request.headers.get("CF-Visitor", "")
        url_scheme = request.url.scheme

        # HTTPS 감지: 여러 방법 시도
        # 환경 변수로 프로덕션 여부 확인
        is_production = os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")
        is_docker = os.path.exists("/.dockerenv")  # Docker 컨테이너 내부인지 확인

        # 로컬 개발 환경 체크 (실제 localhost 접속만)
        is_local_dev = request.url.hostname in ("localhost", "127.0.0.1") or (
            request.url.hostname and "192.168." in request.url.hostname
        )

        # Docker 환경이거나 프로덕션 환경에서는 기본적으로 HTTPS로 가정
        # 로컬 개발 환경이 아니고 헤더가 없으면 HTTPS로 간주
        is_https = (
            x_forwarded_proto == "https"
            or x_forwarded_ssl == "on"
            or (cf_visitor and '"scheme":"https"' in cf_visitor)
            or url_scheme == "https"
            or (
                is_docker or (is_production and not is_local_dev)
            )  # Docker/프로덕션 환경에서는 HTTPS로 간주
        )

        # Cross-Site 요청 지원: HTTPS 환경에서는 SameSite=None; Secure 필수
        # Vercel (snu-kdt-kolon-capstone-project-hzad.vercel.app) → data.taeya.org는 Cross-Site
        # Cross-Site에서 쿠키를 사용하려면 SameSite=None; Secure 필요
        same_site = (
            "none" if is_https else "lax"
        )  # HTTPS = Cross-Site 가능, HTTP = Lax만

        # StreamingResponse의 경우 헤더에 직접 Set-Cookie 추가
        if isinstance(response, StreamingResponse):
            # StreamingResponse는 set_cookie()가 작동하지 않을 수 있으므로
            # 쿠키 문자열을 직접 생성하여 headers에 추가
            secure_flag = "; Secure" if is_https else ""
            # SameSite 값: "none" → "None" (HTTP 표준)
            same_site_value = "None" if same_site == "none" else "Lax"
            cookie_value = (
                f"{self.SESSION_COOKIE_NAME}={session_id}; "
                f"HttpOnly; SameSite={same_site_value}; Max-Age={self.SESSION_TTL}"
                f"{secure_flag}; Path=/"
            )

            # Starlette의 MutableHeaders는 MultiDict처럼 동작
            # 기존 Set-Cookie 헤더 확인 및 추가
            existing_cookie = response.headers.get("Set-Cookie")
            if existing_cookie:
                # 기존 쿠키가 있으면 리스트로 결합 (쉼표로 구분)
                response.headers["Set-Cookie"] = f"{existing_cookie}, {cookie_value}"
            else:
                response.headers["Set-Cookie"] = cookie_value
        else:
            # 일반 Response: set_cookie() 대신 헤더에 직접 설정
            # set_cookie()가 Next.js 프록시를 거칠 때 문제가 있을 수 있으므로
            # StreamingResponse와 동일하게 헤더에 직접 설정
            secure_flag = "; Secure" if is_https else ""
            same_site_value = "None" if same_site == "none" else "Lax"
            cookie_value = (
                f"{self.SESSION_COOKIE_NAME}={session_id}; "
                f"HttpOnly; SameSite={same_site_value}; Max-Age={self.SESSION_TTL}"
                f"{secure_flag}; Path=/"
            )

            # 헤더에 직접 Set-Cookie 추가 (set_cookie() 대신)
            existing_cookie = response.headers.get("Set-Cookie")
            if existing_cookie:
                response.headers["Set-Cookie"] = f"{existing_cookie}, {cookie_value}"
            else:
                response.headers["Set-Cookie"] = cookie_value

        return response
