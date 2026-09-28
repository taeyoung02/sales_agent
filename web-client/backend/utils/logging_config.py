"""
Logging configuration for the application
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from utils.logging_context import SessionIdFilter


def setup_logging(log_dir: str = None, log_level: str = "INFO"):
    """
    로깅 설정

    Args:
        log_dir: 로그 파일 디렉토리 경로 (None이면 자동 감지)
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # 로그 디렉토리 자동 감지 (Docker vs 로컬 개발 환경)
    if log_dir is None:
        if os.path.exists("/backend"):
            # Docker 환경
            log_dir = "/backend/logs"
        else:
            # 로컬 개발 환경
            # backend/logs 디렉토리 사용 (session_log.py와 동일한 경로)
            current_file = Path(__file__).resolve()
            # utils/logging_config.py -> backend/utils/logging_config.py -> backend
            backend_dir = current_file.parent.parent
            log_dir = str(backend_dir / "logs")

    log_path = Path(log_dir)

    # 디렉토리 생성 시도 (실패하면 콘솔 로깅만 사용)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        # 디렉토리 생성 실패 시 경고 출력하고 콘솔 로깅만 사용
        print(f"Warning: Failed to create log directory '{log_dir}': {e}")
        print("Falling back to console logging only.")
        log_path = None

    # Root logger 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()

    # Formatter: 세션 ID 포함
    formatter = logging.Formatter(
        "%(asctime)s [PID:%(process)d] [%(name)s] [SESSION:%(session_id)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # SessionIdFilter 생성
    session_filter = SessionIdFilter()

    # File handler (회전 로그) - 디렉토리 생성에 성공한 경우에만
    if log_path is not None:
        try:
            file_handler = RotatingFileHandler(
                log_path / "app.log",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(session_filter)
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to create file handler: {e}")
            print("Using console logging only.")

    # Console handler (stdout) - 항상 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(session_filter)
    root_logger.addHandler(console_handler)

    # uvicorn, fastapi 등의 로거 레벨 조정 (너무 많은 로그 방지)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    return root_logger
