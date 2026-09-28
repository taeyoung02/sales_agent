"""
Scene API router
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import GetSceneResponse, UploadSceneRequest
from utils.gltf_converter import PLYToGLTFConverter

router = APIRouter(prefix="/api/scene", tags=["scene"])


@router.get("", response_model=GetSceneResponse)
async def get_scene(
    input_path: str = None,
    format: str = "ply",
):
    """
    Get scene endpoint - returns glTF file path or converts PLY to glTF

    Args:
        input_path: Path to PLY file (will be converted) or glTF file (will be returned)
        format: Output format "gltf" or "glb" (default: "gltf")

    Returns:
        Scene response with glTF file path
    """
    try:
        from utils.path_utils import get_project_root

        project_root = get_project_root()

        if not input_path:
            raise HTTPException(
                status_code=400, detail="input_path parameter is required"
            )

        input_file = project_root / input_path

        # Check if input is PLY file - convert to glTF
        if input_file.suffix.lower() == ".ply":
            if not input_file.exists():
                raise HTTPException(
                    status_code=404, detail=f"PLY file not found: {input_path}"
                )

            # Convert PLY to glTF
            # converter = PLYToGLTFConverter()
            # if format == "glb":
            #     output_path = input_file.parent / f"{input_file.stem}.glb"
            # else:
            #     output_path = input_file.parent / f"{input_file.stem}.gltf"

            # result_path = converter.convert(str(input_file), str(output_path), format)
            relative_output = Path(input_file).relative_to(project_root)

            # TODO :: 임시로 ply 파일 넘기도록 허용
            print("relative_output >>>>>>>> ", relative_output)

            # Return HTTP URL path for frontend access
            http_path = f"/static/{relative_output.as_posix()}"

            return GetSceneResponse(
                success=True,
                output_path=http_path,
                message=f"Successfully converted {input_file.name} to {relative_output.name}",
            )

        # Check if input is already glTF/GLB file - return path
        elif input_file.suffix.lower() in [".gltf", ".glb"]:
            if not input_file.exists():
                raise HTTPException(
                    status_code=404, detail=f"glTF file not found: {input_path}"
                )

            relative_path = input_file.relative_to(project_root)

            # Return HTTP URL path for frontend access
            http_path = f"/static/{relative_path.as_posix()}"

            return GetSceneResponse(
                success=True,
                output_path=http_path,
                message=f"Scene file found: {relative_path.name}",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="input_path must be a .ply, .gltf, or .glb file",
            )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene endpoint failed: {str(e)}")
