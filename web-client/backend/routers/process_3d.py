"""
3D Process Pipeline API router
실제 run_full_pipeline.sh를 실행하고 로그를 스트리밍하는 API
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pathlib import Path
import sys
import os
import asyncio
import subprocess
import json
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import logging
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import Process3DRequest, Process3DResponse, Process3DStatusResponse
from utils.path_utils import get_project_root

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/3d-process", tags=["3d-process"])

# 작업 상태 저장 (실제로는 Redis나 DB를 사용해야 함)
job_states: Dict[str, Dict[str, Any]] = {}


def get_pipeline_script_path() -> Path:
    """
    run_full_pipeline.sh 경로 반환

    Docker 환경: /app/3d_simulator/run_full_pipeline.sh
    로컬 환경: {project_root}/web-client/3d_simulator/run_full_pipeline.sh
    """
    import os

    # Docker 환경 경로 (우선순위 1)
    docker_path = Path("/app/3d_simulator/run_full_pipeline.sh")
    if docker_path.exists():
        logger.info(f"Found pipeline script in Docker: {docker_path}")
        return docker_path

    # 로컬 환경 경로
    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")
    local_path = project_root / "web-client" / "3d_simulator" / "run_full_pipeline.sh"
    logger.info(f"Local path: {local_path}")
    if local_path.exists():
        logger.info(f"Found pipeline script in local: {local_path}")
        return local_path

    # 두 경로 모두 실패 - 상세한 디버그 정보 제공
    is_docker = os.path.exists("/backend") or os.path.exists("/app")

    tried_paths = [
        f"  - {docker_path} (Docker)",
        f"  - {local_path} (Local)",
    ]

    tried_paths.append(f"\nDebug info:")
    tried_paths.append(f"  - Current working directory: {os.getcwd()}")
    tried_paths.append(f"  - Project root: {project_root}")
    tried_paths.append(f"  - __file__ location: {Path(__file__).absolute()}")

    # Docker 환경에서 추가 디버깅 정보
    if is_docker:
        tried_paths.append(f"\nDocker environment debug:")
        tried_paths.append(f"  - /app exists: {Path('/app').exists()}")
        if Path("/app").exists():
            try:
                app_contents = [p.name for p in Path("/app").iterdir()]
                tried_paths.append(f"  - /app contents: {app_contents}")
            except Exception as e:
                tried_paths.append(f"  - /app contents: (error reading: {e})")

        tried_paths.append(
            f"  - /app/3d_simulator exists: {Path('/app/3d_simulator').exists()}"
        )
        if Path("/app/3d_simulator").exists():
            try:
                sim_contents = [p.name for p in Path("/app/3d_simulator").iterdir()][
                    :10
                ]
                tried_paths.append(
                    f"  - /app/3d_simulator contents (first 10): {sim_contents}"
                )
            except Exception as e:
                tried_paths.append(
                    f"  - /app/3d_simulator contents: (error reading: {e})"
                )

        tried_paths.append(f"\n  Docker mount check:")
        tried_paths.append(f"    docker-compose.prod.yml should have:")
        tried_paths.append(f"      - ../3d_simulator:/app/3d_simulator:rw")

    # 로컬 환경에서 추가 디버깅 정보
    if not is_docker:
        tried_paths.append(f"\nLocal environment debug:")
        tried_paths.append(f"  - Project root exists: {project_root.exists()}")
        if project_root.exists():
            try:
                root_contents = [p.name for p in project_root.iterdir()]
                tried_paths.append(f"  - Project root contents: {root_contents}")
            except Exception as e:
                tried_paths.append(f"  - Project root contents: (error reading: {e})")

        web_client_dir = project_root / "web-client"
        tried_paths.append(
            f"  - web-client directory exists: {web_client_dir.exists()}"
        )
        if web_client_dir.exists():
            try:
                web_client_contents = [p.name for p in web_client_dir.iterdir()]
                tried_paths.append(f"  - web-client contents: {web_client_contents}")
            except Exception as e:
                tried_paths.append(f"  - web-client contents: (error reading: {e})")

    error_msg = "Pipeline script not found. Tried:\n" + "\n".join(tried_paths)
    error_msg += "\n\nPlease ensure:"
    if is_docker:
        error_msg += "\n  1. In docker-compose.prod.yml, mount 3d_simulator:"
        error_msg += "\n     - ../3d_simulator:/app/3d_simulator:rw"
    else:
        error_msg += "\n  1. 3d_simulator directory exists at:"
        error_msg += f"\n     {local_path.parent}"

    raise FileNotFoundError(error_msg)


def parse_tqdm_output(line: str) -> Optional[Tuple[int, Optional[int], Optional[int]]]:
    """
    tqdm 출력 또는 일반 진행률 출력에서 진행률 정보 추출

    tqdm 출력 형식 예시:
    - 100%|██████████| 1000/1000 [00:10<00:00, 95.23it/s]
    - 50%|█████     | 500/1000 [00:05<00:10, 95.23it/s]
    - 100%|████████████████████████████████████████| 1000/1000 [00:10<00:00, 95.23it/s]

    일반 진행률 출력 형식 예시:
    - Reading camera 1/28
    - Processing 5/10
    - Iteration 1000/30000

    Returns:
        (percent, current, total) 또는 None
    """
    # ANSI escape sequence 제거 (색상 코드 등)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean_line = ansi_escape.sub("", line)

    # tqdm 패턴 1: 퍼센트와 현재/전체 추출
    # 예: "100%|██████████| 1000/1000 [00:10<00:00, 95.23it/s]"
    pattern1 = r"(\d+)%\s*\|\s*[█\s]+\|\s*(\d+)/(\d+)"
    match1 = re.search(pattern1, clean_line)
    if match1:
        percent = int(match1.group(1))
        current = int(match1.group(2))
        total = int(match1.group(3))
        return (percent, current, total)

    # tqdm 패턴 2: 퍼센트만 추출 (현재/전체가 없는 경우)
    # 예: "50%|█████     |"
    pattern2 = r"(\d+)%\s*\|\s*[█\s]+\|"
    match2 = re.search(pattern2, clean_line)
    if match2:
        percent = int(match2.group(1))
        return (percent, None, None)

    # tqdm 패턴 3: 현재/전체만 추출 (퍼센트가 없는 경우)
    # 예: "1000/1000 [00:10<00:00, 95.23it/s]"
    pattern3 = r"(\d+)/(\d+)\s*\["
    match3 = re.search(pattern3, clean_line)
    if match3:
        current = int(match3.group(1))
        total = int(match3.group(2))
        percent = int((current / total) * 100) if total > 0 else 0
        return (percent, current, total)

    # 일반 진행률 패턴: "Reading camera 1/28", "Processing 5/10" 등
    # 단어 뒤에 숫자/숫자 패턴이 있는 경우
    pattern4 = r"(\d+)/(\d+)(?:\s|$)"
    match4 = re.search(pattern4, clean_line)
    if match4:
        current = int(match4.group(1))
        total = int(match4.group(2))
        if total > 0 and current <= total:  # 유효한 진행률인지 확인
            percent = int((current / total) * 100)
            return (percent, current, total)

    return None


def parse_pipeline_log(line: str, job_id: str):
    """파이프라인 로그를 파싱하여 단계 업데이트"""
    line_lower = line.lower()

    # tqdm 출력 감지 및 파싱 (먼저 처리)
    tqdm_info = parse_tqdm_output(line)
    if tqdm_info:
        percent, current, total = tqdm_info
        # 현재 처리 중인 단계에 진행률 적용
        for step_id, step_data in job_states[job_id]["steps"].items():
            if step_data["status"] == "processing":
                # tqdm 진행률을 해당 단계의 진행률로 업데이트
                job_states[job_id]["steps"][step_id]["progress"] = min(percent, 99)
                # tqdm 출력을 로그에 추가 (선택적)
                if current is not None and total is not None:
                    job_states[job_id]["steps"][step_id]["logs"].append(
                        f"[Progress] {current}/{total} ({percent}%)"
                    )
                else:
                    job_states[job_id]["steps"][step_id]["logs"].append(
                        f"[Progress] {percent}%"
                    )
                # 전체 로그에도 추가
                job_states[job_id]["logs"].append(f"[Progress] {percent}%")
                return  # tqdm 출력은 여기서 처리 완료

    # Phase -1: Video Extraction
    if (
        "phase -1" in line_lower
        or "비디오에서 이미지 추출" in line
        or "video to images" in line_lower
        or "video_extract" in line_lower
    ):
        if "video_extract" not in job_states[job_id]["steps"]:
            # 비디오 추출 단계가 없으면 추가 (동적으로 추가)
            job_states[job_id]["steps"]["video_extract"] = {
                "status": "pending",
                "progress": 0,
                "logs": [],
            }

        if job_states[job_id]["steps"]["video_extract"]["status"] == "pending":
            job_states[job_id]["steps"]["video_extract"]["status"] = "processing"
            job_states[job_id]["steps"]["video_extract"]["progress"] = 10
        if "extract" in line_lower or "추출" in line:
            job_states[job_id]["steps"]["video_extract"]["progress"] = 50
        if ("완료" in line or "complete" in line_lower) and (
            "video" in line_lower or "비디오" in line
        ):
            job_states[job_id]["steps"]["video_extract"]["status"] = "success"
            job_states[job_id]["steps"]["video_extract"]["progress"] = 100

    # Phase 0: Image Preprocessing
    elif "phase 0" in line_lower or "image preprocessing" in line_lower:
        if job_states[job_id]["steps"]["preprocess"]["status"] == "pending":
            job_states[job_id]["steps"]["preprocess"]["status"] = "processing"
            job_states[job_id]["steps"]["preprocess"]["progress"] = 10
        if "realesrgan" in line_lower or "esrgan" in line_lower:
            job_states[job_id]["steps"]["preprocess"]["progress"] = 50
        if ("완료" in line or "complete" in line_lower) and "preprocess" in line_lower:
            job_states[job_id]["steps"]["preprocess"]["status"] = "success"
            job_states[job_id]["steps"]["preprocess"]["progress"] = 100

    # Phase 1: COLMAP
    elif "phase 1" in line_lower or "colmap" in line_lower:
        if job_states[job_id]["steps"]["colmap"]["status"] == "pending":
            job_states[job_id]["steps"]["colmap"]["status"] = "processing"
        if "feature_extractor" in line_lower or "특징점 추출" in line:
            job_states[job_id]["steps"]["colmap"]["progress"] = 20
        elif "exhaustive_matcher" in line_lower or "매칭" in line:
            job_states[job_id]["steps"]["colmap"]["progress"] = 40
        elif "mapper" in line_lower or "재구성" in line:
            job_states[job_id]["steps"]["colmap"]["progress"] = 60
        elif "image_undistorter" in line_lower or "왜곡 제거" in line:
            job_states[job_id]["steps"]["colmap"]["progress"] = 80
        elif ("완료" in line or "complete" in line_lower) and (
            "colmap" in line_lower or "phase 1" in line_lower
        ):
            job_states[job_id]["steps"]["colmap"]["status"] = "success"
            job_states[job_id]["steps"]["colmap"]["progress"] = 100

    # Phase 2: Rembg
    elif "phase 2" in line_lower or "rembg" in line_lower or "배경 제거" in line:
        if "processing" in line_lower or "처리" in line:
            job_states[job_id]["steps"]["rembg"]["status"] = "processing"
        elif "완료" in line or "complete" in line_lower:
            job_states[job_id]["steps"]["rembg"]["status"] = "success"
            job_states[job_id]["steps"]["rembg"]["progress"] = 100

    # Phase 3: 3DGS
    elif "phase 3" in line_lower or "3d gaussian" in line_lower or "3dgs" in line_lower:
        if "training" in line_lower or "학습" in line:
            job_states[job_id]["steps"]["3dgs"]["status"] = "processing"
        # 진행률 추출 시도
        if "iteration" in line_lower or "iter" in line_lower:
            # "iteration 1000/30000" 같은 패턴 찾기
            match = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                progress = min(int((current / total) * 100), 99)
                job_states[job_id]["steps"]["3dgs"]["progress"] = progress
        elif "완료" in line or "complete" in line_lower:
            job_states[job_id]["steps"]["3dgs"]["status"] = "success"
            job_states[job_id]["steps"]["3dgs"]["progress"] = 100

    # Phase 3.5: Pruning
    elif (
        "phase 3.5" in line_lower
        or "pruning" in line_lower
        or "point cloud" in line_lower
    ):
        if "processing" in line_lower or "처리" in line:
            job_states[job_id]["steps"]["pruning"]["status"] = "processing"
        elif "완료" in line or "complete" in line_lower:
            job_states[job_id]["steps"]["pruning"]["status"] = "success"
            job_states[job_id]["steps"]["pruning"]["progress"] = 100

    # Phase 4: LangSplat
    elif "phase 4" in line_lower or "langsplat" in line_lower:
        if "preprocess" in line_lower or "처리" in line:
            job_states[job_id]["steps"]["langsplat"]["status"] = "processing"
        elif "완료" in line or "complete" in line_lower:
            job_states[job_id]["steps"]["langsplat"]["status"] = "success"
            job_states[job_id]["steps"]["langsplat"]["progress"] = 100

    # Phase 5: CF3
    elif "phase 5" in line_lower or "cf3" in line_lower:
        if "training" in line_lower or "학습" in line:
            job_states[job_id]["steps"]["cf3"]["status"] = "processing"
        elif "완료" in line or "complete" in line_lower:
            job_states[job_id]["steps"]["cf3"]["status"] = "success"
            job_states[job_id]["steps"]["cf3"]["progress"] = 100

    # 로그를 해당 단계에 추가
    current_processing_step = None
    for step_id, step_data in job_states[job_id]["steps"].items():
        if step_data["status"] == "processing":
            current_processing_step = step_id
            break

    if current_processing_step:
        job_states[job_id]["steps"][current_processing_step]["logs"].append(line)

    # 전체 로그에도 추가
    job_states[job_id]["logs"].append(line)


async def run_pipeline_async(
    job_id: str,
    project_name: str,
    image_path: str,
    use_esrgan_preprocess: bool = True,
    auto_remove_background: bool = True,
    debug_mode: bool = False,
    iterations: int = 15000,
    max_width: int = 3240,
    extract_from_video: bool = False,
    video_path: Optional[str] = None,
    frame_interval: int = 30,
):
    """비동기로 파이프라인 실행 (실시간 로그 파싱)"""
    try:
        pipeline_script = get_pipeline_script_path()

        # 필수 명령어 확인 (identify는 선택사항이지만 권장)
        missing_commands = []
        warnings = []

        for cmd in ["docker", "find"]:
            result = await asyncio.create_subprocess_exec(
                "which",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()
            if result.returncode != 0:
                missing_commands.append(cmd)

        # docker compose 확인 (선택사항 - 컨테이너 내부에서는 확인 불가)
        # 도커 컨테이너 내부에서는 호스트의 Docker에 접근할 수 없으므로 확인 건너뜀
        # 파이프라인 스크립트가 실행될 때 docker compose가 필요하면 그때 에러가 발생할 것
        try:
            docker_compose_result = await asyncio.create_subprocess_exec(
                "docker",
                "compose",
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await docker_compose_result.wait()
            if docker_compose_result.returncode != 0:
                # docker compose가 없으면 docker-compose (하이픈) 시도
                docker_compose_legacy_result = await asyncio.create_subprocess_exec(
                    "docker-compose",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await docker_compose_legacy_result.wait()
                if docker_compose_legacy_result.returncode != 0:
                    warnings.append(
                        "docker compose - 파이프라인 실행 시 필요할 수 있음 (컨테이너 내부에서는 확인 불가)"
                    )
        except (FileNotFoundError, OSError):
            # docker 명령이 없거나 실행할 수 없으면 경고만 표시
            warnings.append(
                "docker compose - 컨테이너 내부에서 확인 불가, 파이프라인 실행 시 필요할 수 있음"
            )

        # identify는 선택사항 (없어도 스크립트는 진행 가능)
        identify_result = await asyncio.create_subprocess_exec(
            "which",
            "identify",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await identify_result.wait()
        if identify_result.returncode != 0:
            warnings.append(
                "identify (ImageMagick) - 이미지 해상도 자동 감지 불가, RealESRGAN 단계 건너뜀"
            )
            parse_pipeline_log(
                f"[WARNING] 'identify' command not found. Image resolution auto-detection will be skipped.",
                job_id,
            )

        if missing_commands:
            error_msg = f"Missing required commands: {', '.join(missing_commands)}"
            if "docker" in missing_commands:
                error_msg += (
                    "\n  - 'docker' is required. Install Docker Desktop or Docker CLI"
                )
            if "docker compose" in missing_commands:
                error_msg += "\n  - 'docker compose' is required. Make sure Docker Desktop is installed and running"
            if "find" in missing_commands:
                error_msg += (
                    "\n  - 'find' is a standard Unix command and should be available"
                )
            job_states[job_id]["status"] = "error"
            job_states[job_id]["error"] = error_msg
            logger.error(f"Pipeline prerequisites check failed: {error_msg}")
            return

        if warnings:
            logger.warning(f"Pipeline warnings: {', '.join(warnings)}")

        # 환경 변수 설정
        env = os.environ.copy()
        env["PROJECT_NAME"] = project_name
        env["ITERATIONS"] = str(iterations)
        env["MAX_WIDTH"] = str(max_width)
        if extract_from_video:
            env["FRAME_INTERVAL"] = str(frame_interval)

        # Docker socket 경로 설정 (컨테이너 내부에서 호스트 Docker 사용)
        # docker-compose.prod.yml에서 /var/run/docker.sock가 마운트되어 있어야 함
        if os.path.exists("/var/run/docker.sock"):
            env["DOCKER_HOST"] = "unix:///var/run/docker.sock"
            logger.info("Docker socket found, pipeline can use docker compose")
        else:
            logger.warning(
                "Docker socket not found. Pipeline may fail if it requires docker compose. "
                "Ensure /var/run/docker.sock is mounted in docker-compose.prod.yml"
            )

        # 파이프라인 스크립트 실행 (실시간 출력 읽기)
        # 입력을 자동으로 제공하기 위해 stdin으로 전달
        # 스크립트 입력 순서:
        # 1. PROJECT_NAME
        # 2. EXTRACT_FROM_VIDEO (y/N)
        # 3. VIDEO_PATH (EXTRACT_FROM_VIDEO가 y인 경우) 또는 IMAGE_PATH (N인 경우)
        # 4. USE_ESRGAN_PREPROCESS
        # 5. AUTO_REMOVE_BACKGROUND
        # 6. DEBUG_MODE
        # 7. CONFIRM (y로 고정)

        if extract_from_video and video_path:
            # 비디오 처리
            input_data = f"{project_name}\ny\n{video_path}\n{'Y' if use_esrgan_preprocess else 'n'}\n{'Y' if auto_remove_background else 'n'}\n{'y' if debug_mode else 'N'}\ny\n"
        else:
            # 이미지 처리
            input_data = f"{project_name}\nN\n{image_path}\n{'Y' if use_esrgan_preprocess else 'n'}\n{'Y' if auto_remove_background else 'n'}\n{'y' if debug_mode else 'N'}\ny\n"

        logger.info(f"Starting pipeline with input:\n{input_data}")

        # bash를 통해 입력을 파이프로 전달
        # stderr도 별도로 캡처하여 에러 메시지 확인
        # 버퍼링 방지를 위해 환경 변수 설정
        env["PYTHONUNBUFFERED"] = "1"

        # stdbuf가 있으면 사용, 없으면 직접 bash 실행
        # 디버깅을 위해 bash -x 옵션 사용 (각 명령 실행 시 출력)
        try:
            # stdbuf 사용 시도
            process = await asyncio.create_subprocess_exec(
                "stdbuf",
                "-oL",  # 라인 버퍼링 (stdout)
                "-eL",  # 라인 버퍼링 (stderr)
                "bash",
                "-x",  # 디버그 모드: 각 명령 실행 시 출력
                str(pipeline_script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(pipeline_script.parent),
            )
        except FileNotFoundError:
            # stdbuf가 없으면 직접 bash 실행
            logger.warning(
                "stdbuf not found, using bash directly (may have buffering issues)"
            )
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-x",  # 디버그 모드: 각 명령 실행 시 출력
                str(pipeline_script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(pipeline_script.parent),
            )

        # 입력 전달 (비동기로)
        async def send_input():
            try:
                process.stdin.write(input_data.encode())
                await process.stdin.drain()
                process.stdin.close()
            except Exception as e:
                logger.error(f"Failed to write input to process: {e}")
                try:
                    process.stdin.close()
                except:
                    pass

        # 입력 전송 태스크 시작
        input_task = asyncio.create_task(send_input())

        # 입력 전송 완료 대기
        await input_task

        # stdout과 stderr를 동시에 읽기
        error_lines = []
        stderr_lines = []

        async def read_stdout():
            """stdout 읽기"""
            buffer = b""
            while True:
                # 작은 청크로 읽어서 실시간성 향상
                chunk = await process.stdout.read(1024)
                if not chunk:
                    # 마지막 버퍼 처리
                    if buffer:
                        line_str = buffer.decode("utf-8", errors="ignore").strip()
                        if line_str:
                            parse_pipeline_log(line_str, job_id)
                            check_for_errors(line_str)
                    break

                buffer += chunk
                # 줄바꿈 문자로 분리
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        parse_pipeline_log(line_str, job_id)
                        check_for_errors(line_str)

        def check_for_errors(line_str: str):
            """에러 체크 헬퍼 함수"""
            line_lower = line_str.lower()
            if any(
                keyword in line_lower
                for keyword in ["error", "failed", "실패", "오류", "❌"]
            ):
                error_lines.append(line_str)
                # 현재 처리 중인 단계를 error로 표시
                for step_id, step_data in job_states[job_id]["steps"].items():
                    if step_data["status"] == "processing":
                        job_states[job_id]["steps"][step_id]["status"] = "error"
                        break

        async def read_stderr():
            """stderr 읽기 - 프로세스 종료 후에도 남은 데이터 읽기"""
            buffer = b""
            try:
                # 프로세스가 종료될 때까지 계속 읽기
                while True:
                    try:
                        # 타임아웃을 사용하되, 더 길게 설정
                        chunk = await asyncio.wait_for(
                            process.stderr.read(1024), timeout=1.0
                        )
                        if not chunk:
                            # EOF에 도달했지만, 프로세스가 아직 실행 중일 수 있음
                            # 프로세스 상태 확인
                            if process.returncode is not None:
                                # 프로세스가 종료됨
                                break
                            # 프로세스가 아직 실행 중이면 잠시 대기 후 재시도
                            await asyncio.sleep(0.1)
                            continue

                        buffer += chunk
                        # \r (carriage return) 처리 - tqdm이 같은 줄을 업데이트할 때 사용
                        # \r이 있으면 이전 줄을 덮어쓰므로, \r로 분리하여 마지막 줄만 처리
                        if b"\r" in buffer:
                            # \r로 분리하여 마지막 줄만 처리
                            lines = buffer.split(b"\r")
                            buffer = lines[-1]  # 마지막 줄만 버퍼에 유지
                            # 이전 줄들 처리
                            for line_bytes in lines[:-1]:
                                if line_bytes:
                                    line_str = line_bytes.decode(
                                        "utf-8", errors="ignore"
                                    ).strip()
                                    if line_str:
                                        process_stderr_line(line_str)

                        # 줄바꿈 문자로 분리
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line_str = line.decode("utf-8", errors="ignore").strip()
                            if line_str:
                                process_stderr_line(line_str)
                    except asyncio.TimeoutError:
                        # 타임아웃 발생 - 프로세스가 아직 실행 중일 수 있음
                        # 프로세스 상태 확인
                        if process.returncode is not None:
                            # 프로세스가 종료됨
                            break
                        # 프로세스가 아직 실행 중이면 계속 읽기
                        continue
            except Exception as e:
                logger.error(f"Error reading stderr: {e}", exc_info=True)
            finally:
                # 마지막 버퍼 처리
                if buffer:
                    line_str = buffer.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        process_stderr_line(line_str)

                # 프로세스가 종료된 후에도 남은 stderr 읽기 시도
                try:
                    while True:
                        remaining = await asyncio.wait_for(
                            process.stderr.read(1024), timeout=0.5
                        )
                        if not remaining:
                            break
                        buffer = remaining
                        # \r 처리
                        if b"\r" in buffer:
                            lines = buffer.split(b"\r")
                            buffer = lines[-1]
                            for line_bytes in lines[:-1]:
                                if line_bytes:
                                    line_str = line_bytes.decode(
                                        "utf-8", errors="ignore"
                                    ).strip()
                                    if line_str:
                                        process_stderr_line(line_str)
                        # 줄바꿈 문자로 분리
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line_str = line.decode("utf-8", errors="ignore").strip()
                            if line_str:
                                process_stderr_line(line_str)
                        if buffer:
                            line_str = buffer.decode("utf-8", errors="ignore").strip()
                            if line_str:
                                process_stderr_line(line_str)
                except (asyncio.TimeoutError, OSError):
                    # 더 이상 읽을 데이터가 없음
                    pass

        def process_stderr_line(line_str: str):
            """stderr 라인 처리"""
            # tqdm 출력 감지 및 처리 (먼저 처리)
            tqdm_info = parse_tqdm_output(line_str)
            if tqdm_info:
                percent, current, total = tqdm_info
                # 현재 처리 중인 단계에 진행률 적용
                for step_id, step_data in job_states[job_id]["steps"].items():
                    if step_data["status"] == "processing":
                        job_states[job_id]["steps"][step_id]["progress"] = min(
                            percent, 99
                        )
                        # tqdm 출력을 로그에 추가하지 않음 (너무 많이 출력되므로)
                        # 대신 진행률만 업데이트
                        return  # tqdm 출력은 여기서 처리 완료, 에러로 처리하지 않음

            # cp 명령의 경고 메시지는 무시 (에러가 아님)
            if any(
                warning in line_str.lower()
                for warning in [
                    "identical (not copied)",
                    "동일한 파일",
                    "are identical",
                    "same file",
                ]
            ):
                # 경고만 로그에 추가하고 에러로 처리하지 않음
                parse_pipeline_log(f"[WARNING] {line_str}", job_id)
                return

            stderr_lines.append(line_str)
            line_lower = line_str.lower()

            # 실제 에러인지 확인 (경고가 아닌)
            is_error = any(
                keyword in line_lower
                for keyword in [
                    "error",
                    "failed",
                    "실패",
                    "오류",
                    "❌",
                    "command not found",
                    "no such file",
                    "cannot",
                    "permission denied",
                    "access denied",
                    "not found",
                    "bash:",
                    "line",
                    "syntax error",
                    "docker:",
                    "unable to",
                    "cannot connect",
                    "connection refused",
                    "exit code",
                    "exit status",
                ]
            )

            # 모든 stderr를 로그에 추가 (디버깅용)
            # bash -x 모드에서는 stderr에 디버그 정보가 출력될 수 있음
            parse_pipeline_log(f"[STDERR] {line_str}", job_id)
            # 로거에도 기록 (디버깅용)
            logger.debug(f"Pipeline stderr: {line_str}")

            if is_error:
                parse_pipeline_log(f"[ERROR] {line_str}", job_id)
                error_lines.append(line_str)
                # 현재 처리 중인 단계를 error로 표시
                for step_id, step_data in job_states[job_id]["steps"].items():
                    if step_data["status"] == "processing":
                        job_states[job_id]["steps"][step_id]["status"] = "error"
                        break

        # stdout과 stderr를 동시에 읽기
        await asyncio.gather(read_stdout(), read_stderr(), return_exceptions=True)

        # 프로세스 완료 대기
        return_code = await process.wait()

        # 프로세스 종료 후 남은 stderr 한 번 더 읽기
        try:
            remaining_stderr = await asyncio.wait_for(
                process.stderr.read(8192), timeout=1.0
            )
            if remaining_stderr:
                buffer = b""
                buffer += remaining_stderr
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        process_stderr_line(line_str)
                if buffer:
                    line_str = buffer.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        process_stderr_line(line_str)
        except (asyncio.TimeoutError, OSError):
            pass

        # 프로세스가 실패했는데 stderr에 에러가 없으면, 모든 stderr를 에러로 표시
        if return_code != 0 and not error_lines and stderr_lines:
            for line_str in stderr_lines[-20:]:  # 최근 20줄만
                # 경고 메시지는 제외
                if any(
                    warning in line_str.lower()
                    for warning in [
                        "identical (not copied)",
                        "동일한 파일",
                        "are identical",
                        "same file",
                        "redis",
                    ]
                ):
                    continue
                error_lines.append(line_str)
                parse_pipeline_log(f"[ERROR] {line_str}", job_id)

        # 프로세스가 실패했는데 stderr도 없으면, 마지막 stdout 라인들을 확인
        if return_code != 0 and not error_lines and not stderr_lines:
            # 마지막 stdout 라인들을 에러로 표시하지 않음 (이미 로그에 있음)
            # 대신 실제 문제를 진단
            pass

        job_states[job_id]["status"] = "completed" if return_code == 0 else "error"
        job_states[job_id]["return_code"] = return_code

        if return_code != 0:
            # 에러 메시지 구성 (최근 에러 라인 포함)
            error_msg = f"Pipeline exited with code {return_code}"

            # 모든 stderr 메시지 포함 (Redis 오류 제외, 최대 50줄)
            if stderr_lines:
                # Redis 관련 오류 필터링
                filtered_stderr = [
                    line
                    for line in stderr_lines
                    if "redis" not in line.lower() and "ConnectionError" not in line
                ]
                if filtered_stderr:
                    # 최근 50줄까지 표시 (더 많은 정보)
                    recent_stderr = "\n".join(filtered_stderr[-50:])
                    error_msg = f"{error_msg}\n\nAll Stderr output (last 50 lines):\n{recent_stderr}"
                    logger.error(f"Pipeline stderr (last 50 lines):\n{recent_stderr}")

            # 에러 라인이 있으면 포함 (Redis 오류 제외)
            if error_lines:
                filtered_errors = [
                    line
                    for line in error_lines
                    if "redis" not in line.lower() and "ConnectionError" not in line
                ]
                if filtered_errors:
                    recent_errors = "\n".join(filtered_errors[-10:])
                    if stderr_lines:
                        error_msg = f"{error_msg}\n\nError lines:\n{recent_errors}"
                    else:
                        error_msg = f"{error_msg}\n\nRecent errors:\n{recent_errors}"

            # 전체 로그의 마지막 부분도 포함 (디버깅용, Redis 오류 제외)
            if job_states[job_id]["logs"]:
                filtered_logs = [
                    log
                    for log in job_states[job_id]["logs"][-30:]
                    if "redis" not in log.lower() and "ConnectionError" not in log
                ]
                if filtered_logs:
                    recent_logs = "\n".join(filtered_logs[-20:])
                    error_msg = f"{error_msg}\n\nRecent logs:\n{recent_logs}"

            # stderr가 없고 에러도 없으면, 스크립트가 조용히 실패한 것
            if not stderr_lines and not error_lines:
                error_msg = f"{error_msg}\n\nNo error messages captured. The script may have failed silently.\n\n"
                error_msg += "Diagnosis:\n"

                # 마지막 로그를 분석하여 어디서 실패했는지 추정
                # bash -x 모드에서 나온 디버그 정보도 포함
                last_logs = [
                    log
                    for log in job_states[job_id]["logs"][-30:]  # 더 많은 로그 확인
                    if "redis" not in log.lower()
                    and not log.startswith("[ERROR]")
                    and not log.startswith("[WARNING]")
                ]

                if last_logs:
                    error_msg += f"Last output (including bash -x debug info):\n"
                    for log in last_logs[-15:]:  # 최근 15줄
                        error_msg += f"  {log}\n"
                    error_msg += "\n"

                    # bash -x 출력에서 마지막 실행된 명령 찾기
                    bash_commands = [log for log in last_logs if log.startswith("+ ")]
                    if bash_commands:
                        error_msg += f"Last executed commands (bash -x):\n"
                        for cmd in bash_commands[-5:]:
                            error_msg += f"  {cmd}\n"
                        error_msg += "\n"

                error_msg += "Possible causes:\n"
                error_msg += "  1. Docker run command failed (check if Docker images exist: 3d-reconstruction-pipeline:latest)\n"
                error_msg += "  2. Script failed at Phase 0.2 (ESRGAN preprocessing) - check docker run output\n"
                error_msg += "  3. Permission issues with file operations\n"
                error_msg += "  4. Missing dependencies in Docker containers\n"
                error_msg += "  5. GPU access issues (--device nvidia.com/gpu=all)\n\n"
                error_msg += "To debug:\n"
                error_msg += "  - Check if Docker images are built: docker images | grep 3d-reconstruction-pipeline\n"
                error_msg += "  - Check Docker daemon is running: docker ps\n"
                error_msg += (
                    "  - Check bash -x output above to see which command failed"
                )

            job_states[job_id]["error"] = error_msg
            logger.error(f"Pipeline failed: {error_msg}")

    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        job_states[job_id]["status"] = "error"
        job_states[job_id]["error"] = str(e)


@router.post("/start", response_model=Process3DResponse)
async def start_process(
    files: List[UploadFile] = File(...),
    project_name: Optional[str] = Form(None),
    use_esrgan_preprocess: str = Form("Y"),
    auto_remove_background: str = Form("Y"),
    debug_mode: str = Form("N"),
    iterations: int = Form(15000),
    max_width: int = Form(3240),
    extract_from_video: str = Form("N"),
    frame_interval: int = Form(30),
):
    """
    3D 프로세스 파이프라인 시작

    Args:
        files: 업로드된 이미지 파일들
        project_name: 프로젝트 이름 (없으면 자동 생성)
        use_esrgan_preprocess: ESRGAN 전처리 사용 여부 (Y/n)
        auto_remove_background: 자동 배경 제거 여부 (Y/n)
        debug_mode: 디버그 모드 (중간 파일 유지) 여부 (Y/N)
        iterations: 3DGS 학습 반복 횟수 (기본값: 15000)
        max_width: 최대 이미지 너비 (기본값: 3240)
        extract_from_video: 비디오에서 이미지 추출 여부 (Y/N)
        frame_interval: 비디오 프레임 간격 (기본값: 30)

    Returns:
        Process3DResponse: 작업 ID 및 SSE URL
    """
    try:
        # 파일 검증
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")

        logger.info(f"Received {len(files)} files for processing")
        for i, file in enumerate(files):
            logger.info(
                f"File {i+1}: {file.filename}, content_type: {file.content_type}"
            )

        # 파일 타입 확인 (첫 번째 파일 기준)
        is_video = False
        if files and files[0].content_type:
            is_video = files[0].content_type.startswith("video/")

        # extract_from_video 옵션 확인
        extract_from_video_bool = extract_from_video.upper() == "Y" or is_video

        # 비디오인 경우 단일 파일만 허용
        if extract_from_video_bool and len(files) > 1:
            raise HTTPException(
                status_code=400, detail="비디오 파일은 하나만 업로드 가능합니다."
            )
        # 작업 ID 생성
        job_id = str(uuid.uuid4())

        # 프로젝트 이름 생성
        if not project_name:
            project_name = f"project_{int(datetime.now().timestamp())}"

        # 작업 상태 초기화 (비디오인 경우 비디오 추출 단계 추가)
        base_steps = {
            "preprocess": {"status": "pending", "progress": 0, "logs": []},
            "colmap": {"status": "pending", "progress": 0, "logs": []},
            "rembg": {"status": "pending", "progress": 0, "logs": []},
            "3dgs": {"status": "pending", "progress": 0, "logs": []},
            "pruning": {"status": "pending", "progress": 0, "logs": []},
            "langsplat": {"status": "pending", "progress": 0, "logs": []},
            "cf3": {"status": "pending", "progress": 0, "logs": []},
        }

        if extract_from_video_bool:
            base_steps = {
                "video_extract": {"status": "pending", "progress": 0, "logs": []},
                **base_steps,
            }

        job_states[job_id] = {
            "status": "processing",
            "project_name": project_name,
            "steps": base_steps,
            "logs": [],
            "error": None,
        }

        # 파일 저장
        # 도커 환경과 로컬 환경 모두 지원
        import os

        # 파이프라인 스크립트 경로를 먼저 확인하여 데이터 디렉토리 위치 결정
        try:
            pipeline_script = get_pipeline_script_path()
            script_dir = pipeline_script.parent
            logger.info(f"Pipeline script found at: {pipeline_script}")
            logger.info(f"Script directory: {script_dir}")
        except FileNotFoundError as e:
            # 스크립트를 찾을 수 없으면 에러 반환
            error_msg = str(e)
            error_msg += "\n\nPlease ensure:\n"
            error_msg += "  1. In Docker: Mount 3d_simulator to /app/3d_simulator in docker-compose.prod.yml\n"
            error_msg += "  2. In local: 3d_simulator directory exists in the project root or parent directory"
            raise HTTPException(status_code=500, detail=error_msg)

        # 스크립트 디렉토리 기준으로 data 디렉토리 찾기
        # 도커: /app/3d_simulator/data
        # 로컬: {script_dir}/data
        data_dir = script_dir / "data"

        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Data directory exists: {data_dir.exists()}")
        logger.info(f"Data directory parent exists: {data_dir.parent.exists()}")

        # data 디렉토리가 없으면 생성
        if not data_dir.exists():
            try:
                # 부모 디렉토리 확인
                if not data_dir.parent.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Parent directory does not exist: {data_dir.parent}. "
                        f"Please ensure 3d_simulator is properly mounted in Docker.",
                    )
                data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created data directory: {data_dir}")
            except PermissionError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Permission denied creating data directory: {data_dir}. "
                    f"Error: {str(e)}. "
                    f"Check Docker volume mount permissions.",
                )
            except OSError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create data directory: {data_dir}. "
                    f"Error: {str(e)}. "
                    f"Parent directory: {data_dir.parent} exists: {data_dir.parent.exists()}",
                )

        input_dir = data_dir / project_name / "input"

        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created input directory: {input_dir}")
        except PermissionError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Permission denied creating input directory: {input_dir}. "
                f"Error: {str(e)}. "
                f"Data directory: {data_dir} exists: {data_dir.exists()}",
            )
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create input directory: {input_dir}. "
                f"Error: {str(e)}. "
                f"Data directory: {data_dir} exists: {data_dir.exists()}",
            )

        saved_files = []
        video_path = None
        image_path = None

        script_dir = pipeline_script.parent

        if extract_from_video_bool:
            # 비디오 파일 저장
            video_dir = data_dir / project_name / "video"
            video_dir.mkdir(parents=True, exist_ok=True)

            file = files[0]
            if not file.filename:
                raise HTTPException(status_code=400, detail="비디오 파일명이 없습니다.")

            video_file_path = video_dir / file.filename
            with open(video_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file.filename)
            logger.info(f"Saved video file: {video_file_path}")

            # 비디오 경로 설정 (스크립트 실행 위치 기준)
            try:
                relative_video_path = video_file_path.relative_to(script_dir)
                video_path = str(relative_video_path)
            except ValueError:
                video_path = str(video_file_path)
                logger.warning(
                    f"Could not create relative path, using absolute: {video_path}"
                )
        else:
            # 이미지 파일 저장
            for file in files:
                if not file.filename:
                    logger.warning(f"Skipping file without filename")
                    continue
                file_path = input_dir / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                saved_files.append(file.filename)
                logger.info(f"Saved file: {file_path}")

            if not saved_files:
                raise HTTPException(status_code=400, detail="No valid files were saved")

            # 이미지 경로 설정 (스크립트가 실행될 위치 기준)
            # 스크립트는 script_dir에서 실행되므로, 상대 경로 사용
            try:
                relative_path = input_dir.relative_to(script_dir)
                image_path = str(relative_path)
            except ValueError:
                # 상대 경로로 변환할 수 없으면 절대 경로 사용
                image_path = str(input_dir)
                logger.warning(
                    f"Could not create relative path, using absolute: {image_path}"
                )

        logger.info(f"Extract from video: {extract_from_video_bool}")
        if extract_from_video_bool:
            logger.info(f"Video path for pipeline: {video_path}")
        else:
            logger.info(f"Image path for pipeline: {image_path}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Pipeline script: {pipeline_script}")
        logger.info(f"Script directory: {script_dir}")

        # 파이프라인 비동기 실행
        asyncio.create_task(
            run_pipeline_async(
                job_id=job_id,
                project_name=project_name,
                image_path=image_path or "dummy",  # 비디오인 경우 dummy 사용
                use_esrgan_preprocess=use_esrgan_preprocess.upper() == "Y",
                auto_remove_background=auto_remove_background.upper() == "Y",
                debug_mode=debug_mode.upper() == "Y",
                iterations=iterations,
                max_width=max_width,
                extract_from_video=extract_from_video_bool,
                video_path=video_path,
                frame_interval=frame_interval,
            )
        )

        return Process3DResponse(
            job_id=job_id,
            status="processing",
            sse_url=f"/api/3d-process/stream/{job_id}",
        )
    except HTTPException:
        # HTTPException은 그대로 전달
        raise
    except ValueError as e:
        # 값 검증 오류는 400으로
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Start process error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start process: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=Process3DStatusResponse)
async def get_status(job_id: str):
    """
    작업 상태 조회

    Args:
        job_id: 작업 ID

    Returns:
        Process3DStatusResponse: 작업 상태 및 로그
    """
    if job_id not in job_states:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job_states[job_id]

    # 단계 정보 변환
    steps = [
        {
            "id": step_id,
            "status": step_data["status"],
            "progress": step_data["progress"],
            "logs": step_data["logs"],
        }
        for step_id, step_data in state["steps"].items()
    ]

    return Process3DStatusResponse(
        job_id=job_id,
        status=state["status"],
        steps=steps,
        logs=state["logs"][-100:],  # 최근 100줄만
        error=state.get("error"),
    )


@router.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    """
    SSE로 실시간 로그 스트리밍

    Args:
        job_id: 작업 ID

    Returns:
        StreamingResponse: SSE 스트림 (text/event-stream)
    """
    if job_id not in job_states:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_log_count = 0
        last_step_updates = {}

        while True:
            if job_id not in job_states:
                break

            state = job_states[job_id]

            # 새 로그 전송
            if len(state["logs"]) > last_log_count:
                new_logs = state["logs"][last_log_count:]
                for log in new_logs:
                    # SSE 형식: event: log\ndata: {...}\n\n
                    yield f"event: log\ndata: {json.dumps({'message': log})}\n\n"
                last_log_count = len(state["logs"])

            # 단계 업데이트 전송
            for step_id, step_data in state["steps"].items():
                step_key = f"{step_id}_{step_data['status']}_{step_data['progress']}"
                if step_key != last_step_updates.get(step_id):
                    # SSE 형식: event: step_update\ndata: {...}\n\n
                    yield f"event: step_update\ndata: {json.dumps({'step_id': step_id, 'status': step_data['status'], 'progress': step_data['progress']})}\n\n"
                    last_step_updates[step_id] = step_key

            # 완료 또는 에러 시 종료
            if state["status"] in ["completed", "error"]:
                # SSE 형식: event: complete\ndata: {...}\n\n
                yield f"event: complete\ndata: {json.dumps({'status': state['status'], 'error': state.get('error')})}\n\n"
                break

            await asyncio.sleep(1)  # 1초마다 체크

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx buffering 방지
        },
    )
