"""
Redis 클라이언트 및 세션 관리 유틸리티

프로덕션 환경에서 세션 데이터를 Redis에 저장하고 TTL 기반으로 자동 만료 관리
"""

import os
import json
import logging
from typing import Optional, Dict, Any
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisSessionManager:
    """
    Redis 기반 세션 관리자

    - 세션 데이터를 Redis에 저장 (key: session:{session_id})
    - TTL 자동 갱신 (모든 API 요청마다)
    - Keyspace notification을 통한 만료 이벤트 감지
    """

    _redis_client: Optional[Redis] = None
    _pubsub_client: Optional[Redis] = None

    # 세션 TTL (초) - 30분 (요구사항)
    SESSION_TTL = int(os.getenv("SESSION_TTL", 60 * 30))

    # Redis 연결 설정
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    @classmethod
    async def get_redis_client(cls) -> Redis:
        """Redis 클라이언트 싱글톤 반환"""
        if cls._redis_client is None:
            cls._redis_client = await redis.from_url(
                f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}",
                password=cls.REDIS_PASSWORD,
                encoding="utf-8",
                decode_responses=True,  # 자동 UTF-8 디코딩
            )
            logger.info(
                f"Redis client initialized: {cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
            )
        return cls._redis_client

    @classmethod
    async def get_pubsub_client(cls) -> Redis:
        """Redis pub/sub 전용 클라이언트 반환 (키 만료 이벤트 구독용)"""
        if cls._pubsub_client is None:
            cls._pubsub_client = await redis.from_url(
                f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}",
                password=cls.REDIS_PASSWORD,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Redis pub/sub client initialized for keyspace notifications")
        return cls._pubsub_client

    @classmethod
    async def close(cls):
        """Redis 연결 종료"""
        if cls._redis_client:
            await cls._redis_client.aclose()
            cls._redis_client = None
        if cls._pubsub_client:
            await cls._pubsub_client.aclose()
            cls._pubsub_client = None
        logger.info("Redis clients closed")

    # -------- 세션 데이터 관리 --------

    @classmethod
    def _session_key(cls, session_id: str) -> str:
        """세션 키 생성: session:{session_id}"""
        return f"session:{session_id}"

    @classmethod
    async def set_session(
        cls,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        세션 데이터 저장 (TTL과 함께)

        Args:
            session_id: 세션 ID
            data: 저장할 데이터 (dict)
            ttl: TTL (초), None이면 기본값(SESSION_TTL) 사용

        Returns:
            성공 여부
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            value = json.dumps(data, ensure_ascii=False)
            ttl = ttl or cls.SESSION_TTL

            await client.setex(key, ttl, value)
            logger.debug(f"Session set: {session_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to set session {session_id}: {e}", exc_info=True)
            return False

    @classmethod
    async def get_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 데이터 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 데이터 (dict) 또는 None
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            value = await client.get(key)

            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
            return None

    @classmethod
    async def extend_session_ttl(
        cls, session_id: str, ttl: Optional[int] = None
    ) -> bool:
        """
        세션 TTL 갱신 (모든 API 요청마다 호출)

        Args:
            session_id: 세션 ID
            ttl: TTL (초), None이면 기본값(SESSION_TTL) 사용

        Returns:
            성공 여부
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            ttl = ttl or cls.SESSION_TTL

            # EXPIRE 명령으로 TTL 갱신
            result = await client.expire(key, ttl)
            if result:
                logger.debug(f"Session TTL extended: {session_id} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"Session key not found for TTL extension: {session_id}")
                return False
        except Exception as e:
            logger.error(
                f"Failed to extend session TTL {session_id}: {e}", exc_info=True
            )
            return False

    @classmethod
    async def delete_session(cls, session_id: str) -> bool:
        """
        세션 삭제

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            result = await client.delete(key)
            logger.info(f"Session deleted: {session_id}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
            return False

    @classmethod
    async def session_exists(cls, session_id: str) -> bool:
        """
        세션 존재 여부 확인

        Args:
            session_id: 세션 ID

        Returns:
            존재 여부
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            return await client.exists(key) > 0
        except Exception as e:
            logger.error(
                f"Failed to check session existence {session_id}: {e}", exc_info=True
            )
            return False

    @classmethod
    async def get_session_ttl(cls, session_id: str) -> int:
        """
        세션 남은 TTL 조회 (초)

        Args:
            session_id: 세션 ID

        Returns:
            남은 TTL (초), -1: 키 없음, -2: TTL 없음
        """
        try:
            client = await cls.get_redis_client()
            key = cls._session_key(session_id)
            return await client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get session TTL {session_id}: {e}", exc_info=True)
            return -1

    # -------- Redis Config 확인 --------

    @classmethod
    async def check_keyspace_notifications(cls) -> bool:
        """
        Redis keyspace notification 설정 확인

        Returns:
            설정 여부 (notify-keyspace-events에 "Ex" 포함)
        """
        try:
            client = await cls.get_redis_client()
            config = await client.config_get("notify-keyspace-events")

            if config:
                value = config.get("notify-keyspace-events", "")
                # "Ex" 또는 "AKE" 등 만료 이벤트를 포함하는지 확인
                has_expired = "E" in value and ("x" in value or "A" in value)

                if has_expired:
                    logger.info(f"✅ Redis keyspace notifications enabled: {value}")
                    return True
                else:
                    logger.warning(
                        f"⚠️  Redis keyspace notifications NOT properly configured: {value}. "
                        f"Expected 'Ex' or 'AKE'. Session expiry events will NOT be detected!"
                    )
                    return False
            else:
                logger.warning("⚠️  Failed to get Redis notify-keyspace-events config")
                return False
        except Exception as e:
            logger.error(f"Failed to check keyspace notifications: {e}", exc_info=True)
            return False

    @classmethod
    async def enable_keyspace_notifications(cls) -> bool:
        """
        Redis keyspace notification 활성화 (Docker 환경에서 자동 설정용)

        Returns:
            성공 여부
        """
        try:
            client = await cls.get_redis_client()
            # "Ex" = Expired events for keys with TTL
            await client.config_set("notify-keyspace-events", "Ex")
            logger.info("✅ Redis keyspace notifications enabled: Ex")
            return True
        except Exception as e:
            logger.error(
                f"Failed to enable keyspace notifications: {e}. "
                f"Please configure Redis manually with 'notify-keyspace-events Ex'",
                exc_info=True,
            )
            return False

    # -------- 분산 락 (Distributed Lock) --------

    @classmethod
    def _lock_key(cls, resource: str) -> str:
        """락 키 생성: lock:{resource}"""
        return f"lock:{resource}"

    @classmethod
    async def acquire_lock(
        cls, resource: str, timeout: int = 10, lock_ttl: int = 30
    ) -> Optional[str]:
        """
        Redis 분산 락 획득 (SET NX 사용)

        여러 워커 프로세스가 동시에 같은 리소스를 처리하는 것을 방지하기 위한 분산 락.
        SET NX (SET if Not eXists)를 사용하여 원자적으로 락을 획득합니다.

        Args:
            resource: 락을 걸 리소스 식별자 (예: "session_log_save:{session_id}")
            timeout: 락 획득 시도 타임아웃 (초)
            lock_ttl: 락의 TTL (초) - 락이 영원히 유지되지 않도록 보장

        Returns:
            락 식별자 (lock_id) - 락 해제 시 사용, None이면 획득 실패
        """
        import uuid
        import asyncio

        client = await cls.get_redis_client()
        lock_key = cls._lock_key(resource)
        lock_id = str(uuid.uuid4())  # 락 소유자 식별자

        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                # SET NX: 키가 없을 때만 설정 (원자적 연산)
                # EX: TTL 설정 (초 단위)
                result = await client.set(lock_key, lock_id, ex=lock_ttl, nx=True)
                if result:
                    # 락 획득 후 즉시 재확인 (race condition 방지)
                    # 짧은 지연 후 실제 Redis 값을 확인
                    await asyncio.sleep(0.001)  # 1ms 대기
                    actual_value = await client.get(lock_key)
                    if actual_value == lock_id:
                        logger.debug(
                            f"🔒 Lock acquired: {resource} (lock_id: {lock_id[:12]}...)"
                        )
                        return lock_id
                    else:
                        # 락을 획득했다고 생각했지만 실제로는 다른 프로세스가 획득
                        logger.debug(
                            f"⚠️  Lock acquisition race condition: {resource}. "
                            f"Expected {lock_id[:12]}..., got {actual_value[:12] if actual_value else None}..."
                        )
                        # 재시도를 위해 루프 계속
                        await asyncio.sleep(0.05)
                        continue

                # 락 획득 실패 - 타임아웃 확인
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.debug(
                        f"⏱️  Lock acquisition timeout: {resource} (timeout: {timeout}s)"
                    )
                    return None

                # 짧은 대기 후 재시도
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Failed to acquire lock for {resource}: {e}", exc_info=True
                )
                return None

    @classmethod
    async def release_lock(cls, resource: str, lock_id: str) -> bool:
        """
        Redis 분산 락 해제

        Lua 스크립트를 사용하여 락 소유자만 락을 해제할 수 있도록 보장합니다.
        (다른 프로세스가 만료된 락을 해제하는 것을 방지)

        Args:
            resource: 락을 걸 리소스 식별자
            lock_id: 락 획득 시 반환된 lock_id

        Returns:
            성공 여부
        """
        client = await cls.get_redis_client()
        lock_key = cls._lock_key(resource)

        try:
            # Lua 스크립트: 락 소유자만 해제 가능
            # KEYS[1] = lock_key, ARGV[1] = lock_id
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = await client.eval(lua_script, 1, lock_key, lock_id)
            if result:
                logger.debug(f"🔓 Lock released: {resource} (lock_id: {lock_id})")
                return True
            else:
                logger.debug(
                    f"⚠️  Lock release failed (not owner or already released): {resource}"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to release lock for {resource}: {e}", exc_info=True)
            return False
