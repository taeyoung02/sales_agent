"""
Pydantic schemas for request/response models
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    vehicle_id: Optional[str] = (
        None  # Current loaded vehicle ID (primary vehicle if multiple are loaded)
    )
    test_mode: Optional[bool] = None  # Override test mode for this request
    participant_id: Optional[str] = None  # 설문 매칭/중복 사용자 처리용
    user_goal_segment: Optional[str] = None  # 사용자 목표 세그먼트


class ChatResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]


class ConvertRequest(BaseModel):
    ply_path: str
    output_path: Optional[str] = None
    format: Optional[str] = "gltf"  # "gltf" or "glb"


class ConvertResponse(BaseModel):
    success: bool
    output_path: str
    message: str


class GetSceneResponse(BaseModel):
    success: bool
    output_path: str
    message: str


class UploadSceneRequest(BaseModel):
    input_path: str
    format: Optional[str] = "gltf"  # "gltf" or "glb"


class HeatmapRequest(BaseModel):
    query: str
    cf3_path: Optional[str] = None  # Path to CF3 PLY file
    reference_path: Optional[str] = None  # Path to reference PLY file
    vehicle_id: Optional[str] = None  # Optional vehicle ID for DB storage
    use_autoencoder: Optional[bool] = (
        None  # Whether to use autoencoder (None = auto-detect from env)
    )


class HeatmapResponse(BaseModel):
    success: bool
    output_path: str
    message: str
    target_position: Optional[List[float]] = None
    surface_normal: Optional[List[float]] = None
    camera_position: Optional[List[float]] = None
    surface_center: Optional[List[float]] = None


class LoadDataRequest(BaseModel):
    data_dir: Optional[str] = None  # Directory containing car_info*_vector.json files
    clear_existing: bool = False  # Whether to clear existing collection before loading


class LoadDataResponse(BaseModel):
    success: bool
    message: str
    points_loaded: int
    total_points: Optional[int] = None  # Total points in collection after loading


class DataStatusResponse(BaseModel):
    collection_name: str
    total_points: int
    status: str


class PresentationAction(BaseModel):
    type: str  # "setCamera" | "zoomCamera" | "rotateCamera" | "generateHeatmap"
    viewer_id: Optional[str] = None  # 특정 뷰어에만 적용 (viewerId 우선)
    vehicle_id: Optional[str] = None  # 특정 차량에만 적용 (viewer_id가 없을 때 사용)
    preset: Optional[str] = None
    position: Optional[List[float]] = None
    direction: Optional[str] = None
    level: Optional[int] = None
    target: Optional[List[float]] = None
    query: Optional[str] = None
    duration: Optional[float] = None
    delay: Optional[float] = None


class PresentationSegment(BaseModel):
    timestamp: float
    text: str
    actions: List[PresentationAction]


class PresentationScript(BaseModel):
    vehicle_id: str
    total_duration: float
    segments: List[PresentationSegment]


class TTSStreamRequest(BaseModel):
    segments: List[PresentationSegment]
    voice: Optional[str] = "alloy"  # alloy, echo, fable, onyx, nova, shimmer, etc.
    model: Optional[str] = "gpt-4o-mini-tts"  # gpt-4o-mini-tts, tts-1 or tts-1-hd
    speed: Optional[float] = 1.0  # 0.25 to 4.0
    instructions: Optional[str] = None  # Instructions for voice tone, emotion, etc.


class BatchExportRequest(BaseModel):
    """배치 export 요청 스키마"""

    session_ids: Optional[List[str]] = None  # None이면 모든 세션 export


class BatchSaveRequest(BaseModel):
    """배치 파일 저장 요청 스키마"""

    session_ids: Optional[List[str]] = None  # None이면 모든 세션 저장
    delete_after_save: bool = False  # 저장 후 메모리에서 삭제할지 여부


class Process3DRequest(BaseModel):
    """3D 프로세스 시작 요청 스키마"""

    project_name: Optional[str] = None
    use_esrgan_preprocess: bool = True
    auto_remove_background: bool = True
    iterations: int = 30000


class Process3DResponse(BaseModel):
    """3D 프로세스 시작 응답 스키마"""

    job_id: str
    status: str
    sse_url: Optional[str] = None


class ProcessStepStatus(BaseModel):
    """프로세스 단계 상태"""

    id: str
    status: str  # "pending" | "processing" | "success" | "error"
    progress: int  # 0-100
    logs: List[str]


class Process3DStatusResponse(BaseModel):
    """3D 프로세스 상태 응답 스키마"""

    job_id: str
    status: str  # "processing" | "completed" | "error"
    steps: List[ProcessStepStatus]
    logs: List[str]
    error: Optional[str] = None
