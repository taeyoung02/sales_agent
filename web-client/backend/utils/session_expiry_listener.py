"""
Redis keyspace notification 리스너

세션 만료 이벤트(__keyevent@0__:expired)를 감지하여 자동으로 세션 로그 저장
"""

import asyncio
import logging
from typing import Optional
from redis.asyncio import Redis
from utils.redis_client import RedisSessionManager
from utils.session_log import SessionLogStore

logger = logging.getLogger(__name__)


class SessionExpiryListener:
    """
    Redis keyspace notification을 통해 세션 만료 이벤트를 감지하고
    SessionLogStore를 통해 로그 파일로 저장
    """

    _task: Optional[asyncio.Task] = None
    _running = False

    @classmethod
    async def start(cls):
        """백그라운드 태스크 시작"""
        if cls._running:
            logger.warning("SessionExpiryListener is already running")
            return

        cls._running = True
        cls._task = asyncio.create_task(cls._listen_loop())
        logger.info("✅ SessionExpiryListener started")

    @classmethod
    async def stop(cls):
        """백그라운드 태스크 종료"""
        if not cls._running:
            return

        cls._running = False
        if cls._task:
            cls._task.cancel()
            try:
                await cls._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 SessionExpiryListener stopped")

    @classmethod
    async def _listen_loop(cls):
        """
        Redis pub/sub로 키 만료 이벤트 구독

        채널: __keyevent@0__:expired
        메시지 형식: "session:{session_id}"
        """
        pubsub_client = None
        pubsub = None

        try:
            # pub/sub 전용 클라이언트 가져오기
            pubsub_client = await RedisSessionManager.get_pubsub_client()
            pubsub = pubsub_client.pubsub()

            # keyspace notification 채널 구독
            # __keyevent@0__:expired = DB 0에서 만료된 키 이벤트
            await pubsub.subscribe("__keyevent@0__:expired")
            logger.info(
                "📡 Subscribed to Redis keyspace notifications: __keyevent@0__:expired"
            )

            # 무한 루프로 메시지 수신
            while cls._running:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )

                    if message and message["type"] == "message":
                        expired_key = message["data"]
                        logger.debug(f"📩 Received expiration event: {expired_key}")

                        # session:{session_id} 형태의 키만 처리
                        if expired_key.startswith("session:"):
                            session_id = expired_key.replace("session:", "")
                            await cls._handle_session_expiry(session_id)

                    # CPU 과부하 방지를 위한 짧은 대기
                    await asyncio.sleep(0.1)

                except asyncio.TimeoutError:
                    # 타임아웃은 정상 (1초마다 확인)
                    continue
                except asyncio.CancelledError:
                    logger.info("SessionExpiryListener task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in listen loop: {e}", exc_info=True)
                    # 에러 발생 시 잠시 대기 후 재시도
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Fatal error in SessionExpiryListener: {e}", exc_info=True)
        finally:
            # pub/sub 연결 종료 (Redis 연결이 이미 끊어져 있을 수 있으므로 예외 무시)
            if pubsub:
                try:
                    await pubsub.unsubscribe("__keyevent@0__:expired")
                except Exception as e:
                    logger.debug(
                        f"Error unsubscribing from Redis (expected during shutdown): {e}"
                    )
                try:
                    await pubsub.close()
                except Exception as e:
                    logger.debug(
                        f"Error closing pubsub connection (expected during shutdown): {e}"
                    )
            logger.info("SessionExpiryListener loop ended")

    @classmethod
    async def _handle_session_expiry(cls, session_id: str):
        """
        세션 만료 처리: 세션 로그를 파일로 저장

        여러 워커 프로세스가 동시에 같은 세션 만료 이벤트를 처리하는 것을 방지하기 위해
        Redis 분산 락을 사용합니다.

        Args:
            session_id: 만료된 세션 ID
        """
        # 분산 락 리소스 식별자
        lock_resource = f"session_log_save:{session_id}"
        lock_id = None

        try:
            # 만료 이벤트 수신 로그 (락 획득 전)
            logger.info(
                f"📩 Session expiry event received: {session_id}. "
                f"Attempting to acquire lock..."
            )

            # 분산 락 획득 (여러 워커 중 하나만 처리)
            lock_id = await RedisSessionManager.acquire_lock(
                resource=lock_resource, timeout=5, lock_ttl=30
            )

            if not lock_id:
                # 락 획득 실패 = 다른 워커가 이미 처리 중
                logger.info(
                    f"🔒 Lock not acquired for session {session_id}. "
                    f"Another worker is handling this session. Skipping..."
                )
                return

            # 락 획득 후 실제로 Redis에서 확인 (race condition 방지)
            # 여러 프로세스가 동시에 SET NX를 호출해 모두 성공할 수 있으므로 재확인 필요
            import asyncio

            await asyncio.sleep(0.01)  # 짧은 지연으로 네트워크 레이턴시 고려

            from utils.redis_client import RedisSessionManager as RSM

            client = await RSM.get_redis_client()
            # _lock_key() 메서드를 사용하여 정확한 키 생성
            lock_key = RSM._lock_key(lock_resource)
            actual_lock_value = await client.get(lock_key)

            if actual_lock_value != lock_id:
                # 락을 획득했다고 생각했지만 실제로는 다른 프로세스가 획득
                logger.info(
                    f"🔒 Lock verification failed for session {session_id}. "
                    f"Expected lock_id={lock_id[:12]}..., "
                    f"got actual={actual_lock_value[:12] if actual_lock_value else None}... "
                    f"Another worker acquired the lock. Skipping..."
                )
                # 락 해제 (실제로는 우리가 획득하지 않았으므로 해제 불필요하지만 안전을 위해)
                return

            # 락 획득 성공 - 이제 로그 저장 시작
            logger.info(
                f"🔐 Lock acquired and verified for session {session_id} "
                f"(lock_id: {lock_id[:12]}...). Checking for session log data..."
            )

            # SessionLogStore에서 세션 로그 가져오기
            log_data = SessionLogStore.to_dict(session_id)

            if not log_data:
                # 세션 로그가 없음 - 이미 저장되었거나 생성되지 않았을 수 있음
                # vanilla 챗봇처럼 /api/chat 엔드포인트를 호출하지 않은 경우
                logger.warning(
                    f"⚠️  No session log data found for expired session: {session_id}. "
                    f"Possible reasons: "
                    f"1) Session was never used with /api/chat or /api/chat/stream endpoints, "
                    f"2) Session log was already saved and deleted from memory, "
                    f"3) Session expired before any conversation data was recorded."
                )
                return

            # 파일 저장 전에 락을 다시 확인 (다른 프로세스가 이미 저장했을 수 있음)
            # 짧은 지연 후 재확인
            await asyncio.sleep(0.01)
            current_lock_value = await client.get(lock_key)
            if current_lock_value != lock_id:
                # 락을 잃었음 - 다른 프로세스가 이미 처리 중일 수 있음
                logger.info(
                    f"🔒 Lock lost before file save for session {session_id}. "
                    f"Expected {lock_id[:12]}..., got {current_lock_value[:12] if current_lock_value else None}... "
                    f"Another worker may have already saved. Skipping..."
                )
                return

            # 데이터를 가져온 후 즉시 메모리에서 삭제 (다른 프로세스가 중복 저장하는 것 방지)
            # 이렇게 하면 다른 프로세스가 락을 획득해도 데이터가 없어서 저장하지 못함
            SessionLogStore.delete(session_id)
            logger.debug(
                f"🗑️  Session log data removed from memory for {session_id} "
                f"to prevent duplicate saves by other workers"
            )

            # 파일 저장용 추가 락 획득 (파일 저장 단계에서도 중복 방지)
            # participant_id 기반 병합 시 파일명이 달라질 수 있으므로 session_id 기반으로 락 생성
            file_lock_resource = f"file_save:{session_id}"
            logger.info(
                f"🔐 Attempting to acquire file save lock for session {session_id}..."
            )
            file_lock_id = await RedisSessionManager.acquire_lock(
                resource=file_lock_resource, timeout=2, lock_ttl=10
            )

            if not file_lock_id:
                # 파일 저장 락 획득 실패 - 다른 프로세스가 이미 저장 중
                logger.info(
                    f"🔒 File save lock not acquired for session {session_id}. "
                    f"Another worker is saving the file. Skipping..."
                )
                return

            logger.info(
                f"🔐 File save lock acquired for session {session_id} "
                f"(lock_id: {file_lock_id[:12]}...)"
            )

            try:
                # 파일 저장 전에 파일이 이미 존재하는지 확인 (중복 저장 방지)
                # participant_id와 session_id가 모두 같은 경우에만 병합되므로,
                # 기본 파일명은 session_id 기반
                expected_filename = f"session_log_{session_id}.json"
                expected_filepath = SessionLogStore.LOG_DIR / expected_filename

                # 파일이 이미 존재하고 비어있지 않으면 스킵
                if expected_filepath.exists():
                    try:
                        import os

                        file_size = os.path.getsize(expected_filepath)
                        if file_size > 0:
                            logger.info(
                                f"📄 File already exists for session {session_id}: {expected_filename} "
                                f"(size: {file_size} bytes). Skipping save to prevent duplicate."
                            )
                            return
                    except Exception as e:
                        logger.warning(
                            f"Failed to check existing file for session {session_id}: {e}. "
                            f"Proceeding with save..."
                        )

                # 파일로 저장
                logger.info(
                    f"💾 Saving session log to file for session {session_id} "
                    f"(conversation turns: {len(log_data.get('conversation', []))})"
                )
                filepath = SessionLogStore.save_to_file(session_id, log_data)
            finally:
                # 파일 저장 락 해제
                await RedisSessionManager.release_lock(file_lock_resource, file_lock_id)
                logger.info(f"🔓 File save lock released for session {session_id}")

            if filepath:
                logger.info(
                    f"✅ Session log saved on expiry: {session_id} -> {filepath}"
                )
                # 저장 성공 후 즉시 메모리에서 삭제 (다른 프로세스가 중복 저장하는 것 방지)
                SessionLogStore.delete(session_id)
            else:
                logger.error(
                    f"❌ Failed to save session log for expired session: {session_id}"
                )

        except Exception as e:
            logger.error(
                f"Error handling session expiry for {session_id}: {e}",
                exc_info=True,
            )
        finally:
            # 락 해제 (획득에 성공한 경우에만)
            if lock_id:
                await RedisSessionManager.release_lock(lock_resource, lock_id)
                logger.debug(f"🔓 Lock released for session {session_id}")
