"""
FastAPI application entry point for Interactive AI Dealer backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
import asyncio
import logging
from routers import (
    chat,
    scene,
    convert,
    heatmap,
    data,
    prompts,
    camera_presets,
    vanilla_chat,
    process_3d,
)
from utils.path_utils import get_project_root, get_source_root, get_source_file_path
from utils.heatmap_generator import load_clip_encoder, load_autoencoder
from utils.logging_config import setup_logging
from utils.redis_client import RedisSessionManager
from utils.session_expiry_listener import SessionExpiryListener
from middleware.session import SessionMiddleware
import traceback

# Load environment variables
load_dotenv()

# Setup logging
# log_dir이 None이면 자동 감지 (Docker vs 로컬 개발 환경)
log_dir = os.getenv("LOG_DIR", None)  # None이면 자동 감지
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_dir=log_dir, log_level=log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Handles startup and shutdown events
    """
    # Startup: Preload models for faster heatmap generation
    logger.debug("=" * 60)
    logger.debug("🚀 Starting up Interactive AI Dealer backend...")

    # -------- Redis 연결 및 설정 --------
    try:
        logger.info("🔌 Connecting to Redis...")
        # Redis 연결 테스트
        redis_client = await RedisSessionManager.get_redis_client()
        await redis_client.ping()
        logger.info("✅ Redis connection successful")

        # Keyspace notification 설정 확인 및 활성화
        has_notifications = await RedisSessionManager.check_keyspace_notifications()
        if not has_notifications:
            logger.warning("⚠️  Attempting to enable Redis keyspace notifications...")
            enabled = await RedisSessionManager.enable_keyspace_notifications()
            if enabled:
                logger.info("✅ Redis keyspace notifications enabled")
            else:
                logger.error(
                    "❌ Failed to enable Redis keyspace notifications. "
                    "Session expiry events will NOT be detected! "
                    "Please configure Redis manually with 'notify-keyspace-events Ex'"
                )

        # 세션 만료 리스너 시작
        logger.info("📡 Starting session expiry listener...")
        await SessionExpiryListener.start()
        logger.info("✅ Session expiry listener started")
    except Exception as e:
        logger.error(f"❌ Redis setup failed: {e}", exc_info=True)
        logger.warning("⚠️  Backend will run in degraded mode (Redis unavailable)")

    # Preload CLIP encoder (always needed)
    try:
        logger.debug("📦 Preloading CLIP encoder...")
        clip = load_clip_encoder("cpu")
        logger.debug("✅ CLIP encoder preloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Failed to preload CLIP encoder: {e}")

    try:
        source_root = get_source_root()
        ae_model_path = os.getenv(
            "AUTOENCODER_PATH",
            str(source_root / "cf3_demo" / "autoencoder.pth"),
        )
        ae_file = get_source_file_path(ae_model_path, source_root)

        if ae_file.exists():
            logger.debug("📦 Preloading autoencoder model...")
            ae = load_autoencoder(str(ae_file), feature_size=512)
            logger.debug("✅ Autoencoder model preloaded successfully")
        else:
            logger.warning(f"⚠️  Autoencoder file not found: {ae_file}")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Failed to preload autoencoder: {e}")

    logger.info("=" * 60)
    logger.info(f"TEST_MODE: {os.getenv('TEST_MODE')}")
    logger.info(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    logger.info("✅ Backend startup complete")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown: Cleanup
    logger.info("🛑 Shutting down backend...")

    # 세션 만료 리스너 종료
    try:
        logger.info("📡 Stopping session expiry listener...")
        await SessionExpiryListener.stop()
        logger.info("✅ Session expiry listener stopped")
    except Exception as e:
        logger.error(f"Error stopping session expiry listener: {e}", exc_info=True)

    # Redis 연결 종료
    try:
        logger.info("🔌 Closing Redis connections...")
        await RedisSessionManager.close()
        logger.info("✅ Redis connections closed")
    except Exception as e:
        logger.error(f"Error closing Redis connections: {e}", exc_info=True)

    logger.info("✅ Backend shutdown complete")


app = FastAPI(
    title="Interactive AI Dealer API",
    description="Backend API for 3D vehicle simulator with RAG chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (must be added before other middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://192.168.0.13",
        "http://192.168.0.13:3000",
        "http://100.93.57.71",
        "http://100.93.57.71:3000",
        "https://snu-kdt-kolon-capstone-project.vercel.app",
    ],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware (쿠키 기반 세션 ID 관리 및 Redis TTL 갱신)
# IMPORTANT: Nginx 프록시 뒤에서 동작 (proxy_set_header Host $host 필수)
app.add_middleware(SessionMiddleware)

# Mount source directory separately for easier access
# In Docker container, source is mounted at /backend/source
# Check if container path exists, otherwise use get_source_root() for local development
import os

if os.path.exists("/backend/source"):
    source_root = Path("/backend/source")
else:
    source_root = get_source_root()
app.mount("/static/source", StaticFiles(directory=str(source_root)), name="source")

# Include routers
app.include_router(chat.router)
app.include_router(scene.router)
app.include_router(convert.router)
app.include_router(heatmap.router)
app.include_router(data.router)
app.include_router(prompts.router)
app.include_router(camera_presets.router)
app.include_router(vanilla_chat.router)
app.include_router(process_3d.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Interactive AI Dealer API", "version": "0.1.0"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "backend"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
