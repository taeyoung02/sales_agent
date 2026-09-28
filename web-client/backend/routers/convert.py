"""
File conversion API router
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import ConvertRequest, ConvertResponse
from utils.splat_converter import PLYToKSplatConverter
from utils.gltf_converter import PLYToGLTFConverter

router = APIRouter(prefix="/api/convert", tags=["convert"])


@router.post("/ply-to-ksplat", response_model=ConvertResponse)
async def convert_ply_to_ksplat(request: ConvertRequest):
    """
    Convert PLY file to KSplat format

    Args:
        request: Convert request with PLY file path and optional output path

    Returns:
        Convert response with output path and status
    """
    try:
        from utils.path_utils import get_project_root

        project_root = get_project_root()
        ply_path = project_root / request.ply_path

        if not ply_path.exists():
            raise HTTPException(
                status_code=404, detail=f"PLY file not found: {request.ply_path}"
            )

        # Generate output path if not provided
        if request.output_path:
            output_path = project_root / request.output_path
        else:
            output_path = ply_path.parent / f"{ply_path.stem}.ksplat"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert PLY to KSplat
        converter = PLYToKSplatConverter()
        result_path = converter.convert(str(ply_path), str(output_path))

        # Return relative path from project root
        relative_output = Path(result_path).relative_to(project_root)

        return ConvertResponse(
            success=True,
            output_path=str(relative_output),
            message=f"Successfully converted {ply_path.name} to {relative_output.name}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/ply-to-gltf", response_model=ConvertResponse)
async def convert_ply_to_gltf(request: ConvertRequest):
    """
    Convert PLY file to glTF/GLB format

    Args:
        request: Convert request with PLY file path, optional output path, and format

    Returns:
        Convert response with output path and status
    """
    try:
        from utils.path_utils import get_project_root

        project_root = get_project_root()
        ply_path = project_root / request.ply_path

        if not ply_path.exists():
            raise HTTPException(
                status_code=404, detail=f"PLY file not found: {request.ply_path}"
            )

        # Validate format
        format = request.format or "gltf"
        if format not in ["gltf", "glb"]:
            raise HTTPException(
                status_code=400, detail="Format must be 'gltf' or 'glb'"
            )

        # Generate output path if not provided
        if request.output_path:
            output_path = project_root / request.output_path
        else:
            if format == "glb":
                output_path = ply_path.parent / f"{ply_path.stem}.glb"
            else:
                output_path = ply_path.parent / f"{ply_path.stem}.gltf"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert PLY to glTF/GLB
        converter = PLYToGLTFConverter()
        result_path = converter.convert(str(ply_path), str(output_path), format)

        # Return relative path from project root
        relative_output = Path(result_path).relative_to(project_root)

        return ConvertResponse(
            success=True,
            output_path=str(relative_output),
            message=f"Successfully converted {ply_path.name} to {relative_output.name}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
