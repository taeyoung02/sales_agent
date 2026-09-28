"""
Heatmap generation API router
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from pathlib import Path
import os
import uuid
import asyncio
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.schemas import HeatmapRequest, HeatmapResponse
from utils.heatmap_generator import (
    generate_heatmap,
    generate_heatmap_target_position_only,
    load_query_preset,
    save_query_preset,
)
from utils.path_utils import get_source_file_path, get_source_root, get_project_root
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/api/heatmap", tags=["heatmap"])


def get_heatmap_filename(query: str) -> str:
    safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{safe_query}_heatmap.ply"


def get_heatmap_paths(query: str, output_dir: Path) -> tuple[Path, Path]:
    """
    Get paths for heatmap PLY file and metadata JSON file

    Args:
        query: Text query for heatmap
        output_dir: Output directory for heatmaps

    Returns:
        Tuple of (ply_path, metadata_path)
    """
    filename = get_heatmap_filename(query)
    ply_path = output_dir / filename
    metadata_path = output_dir / f"{Path(filename).stem}.json"
    return ply_path, metadata_path


@router.post("/generate", response_model=HeatmapResponse)
async def generate_heatmap_endpoint(request: HeatmapRequest):
    """
    Generate heatmap PLY file based on text query using CF3 feature vectors

    Args:
        request: Heatmap request with query and optional paths

    Returns:
        Heatmap response with output PLY file path
    """
    # DB is optional - we'll try to use it only if vehicle_id is provided
    db = None

    try:
        source_root = get_source_root()
        project_root = get_project_root()

        # Determine whether to use autoencoder
        # Priority: request.use_autoencoder > USE_AUTOENCODER env var > default True
        use_autoencoder = request.use_autoencoder
        if use_autoencoder is None:
            use_autoencoder_env = os.getenv("USE_AUTOENCODER", "true").lower()
            use_autoencoder = use_autoencoder_env in ("true", "1", "yes")

        print(f"Autoencoder usage: {use_autoencoder}")

        # If vehicle_id is provided, use it to determine paths
        if request.vehicle_id:
            vehicle_dir = source_root / request.vehicle_id
            # Default paths for vehicle-specific CF3 files
            # Use feature_field.ply for CF3 (contains 3D features), point_cloud.ply for reference
            default_cf3_path = vehicle_dir / "feature_field.ply"
            default_reference_path = vehicle_dir / "point_cloud.ply"
            default_ae_path = vehicle_dir / "autoencoder.pth"

            # Use vehicle-specific paths if files exist, otherwise fall back to defaults
            cf3_path = request.cf3_path or (
                str(default_cf3_path)
                if default_cf3_path.exists()
                else os.getenv(
                    "CF3_PLY_PATH", str(source_root / "test" / "point_cloud.ply")
                )
            )
            reference_path = request.reference_path or (
                str(default_reference_path)
                if default_reference_path.exists()
                else os.getenv(
                    "REFERENCE_PLY_PATH", str(source_root / "test" / "point_cloud.ply")
                )
            )
            if use_autoencoder:
                # vehicle_id가 있을 때는 해당 vehicle 디렉토리의 autoencoder를 우선 사용
                # CF3와 reference 파일과 동일한 로직 적용
                ae_model_path = (
                    str(default_ae_path)
                    if default_ae_path.exists()
                    else str(
                        default_ae_path
                    )  # vehicle_id 디렉토리 기반 경로 사용 (환경변수 무시)
                )
        else:
            # Get paths from environment or use defaults (original logic)
            cf3_path = request.cf3_path or os.getenv(
                "CF3_PLY_PATH",
                str(source_root / "test" / "cf3.ply"),
            )
            reference_path = request.reference_path or os.getenv(
                "REFERENCE_PLY_PATH",
                str(source_root / "test" / "cf3.ply"),
            )
            # Only get autoencoder path if using autoencoder
            if use_autoencoder:
                ae_model_path = os.getenv(
                    "AUTOENCODER_PATH",
                    str(source_root / "cf3_demo" / "autoencoder.pth"),
                )

        # Resolve absolute paths
        cf3_file = get_source_file_path(cf3_path, source_root)
        ref_file = get_source_file_path(reference_path, source_root)
        # 무조건 autoencoder.pth 파일을 사용
        ae_file = get_source_file_path(ae_model_path, source_root)

        print(f"Resolved CF3 file: {cf3_file}")
        print(f"Resolved reference file: {ref_file}")
        print(f"Resolved autoencoder file: {ae_file}")

        # Validate files exist
        if not cf3_file.exists():
            error_msg = (
                f"CF3 PLY file not found: {cf3_file}\n"
                f"Expected path: {cf3_path}\n"
                f"Please set CF3_PLY_PATH environment variable or provide cf3_path in the request.\n"
                f"Example: export CF3_PLY_PATH='path/to/point_cloud.ply'"
            )
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        if not ref_file.exists():
            error_msg = (
                f"Reference PLY file not found: {ref_file}\n"
                f"Expected path: {reference_path}\n"
                f"Please set REFERENCE_PLY_PATH environment variable or provide reference_path in the request."
            )
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        if not ae_file.exists():
            error_msg = (
                f"Autoencoder model not found: {ae_file}\n"
                f"Expected path: {ae_model_path}\n"
                f"Please set AUTOENCODER_PATH environment variable.\n"
                f"Example: export AUTOENCODER_PATH='path/to/autoencoder.pth'"
            )
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)

        # Create output directory for heatmaps
        # If vehicle_id is provided, use vehicle-specific heatmaps directory
        if request.vehicle_id:
            output_dir = source_root / request.vehicle_id / "heatmaps"
        else:
            # Otherwise, use heatmaps subfolder in the same directory as cf3_file
            output_dir = cf3_file.parent / "heatmaps"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using output directory: {output_dir}")
        if request.vehicle_id:
            print(f"Vehicle ID: {request.vehicle_id}")

            # Auto-generate mean_vector.pt if it doesn't exist
            dataset_path = source_root / request.vehicle_id
            mean_vector_file = dataset_path / "mean_vector.pt"

            if not mean_vector_file.exists():
                print(
                    f"⚠️  mean_vector.pt not found, will auto-generate during heatmap creation"
                )

        # Check if heatmap metadata JSON file already exists
        heatmap_ply_path, heatmap_metadata_path = get_heatmap_paths(
            request.query, output_dir
        )

        # Check if PLY generation is enabled (development mode only)
        # In production, we only generate JSON metadata for camera positioning
        generate_ply = os.getenv("HEATMAP_GENERATE_PLY", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Check if metadata JSON exists (primary source for production)
        metadata_exists = heatmap_metadata_path.exists()
        ply_exists = heatmap_ply_path.exists()

        # If metadata exists and PLY generation is disabled, use existing metadata
        # If metadata exists but PLY generation is enabled and PLY doesn't exist, generate PLY
        # If metadata exists and PLY exists and PLY generation is enabled, use existing files
        if metadata_exists and not generate_ply:
            # Production mode: Use existing metadata, no PLY needed
            print(f"✅ Using existing heatmap metadata: {heatmap_metadata_path.name}")

            # Load metadata from JSON file
            target_position = None
            surface_normal = None
            camera_position = None
            surface_center = None
            try:
                with open(heatmap_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    target_position = metadata.get("target_position")
                    surface_normal = metadata.get("surface_normal")
                    camera_position = metadata.get("camera_position")
                    surface_center = metadata.get("surface_center")
                print(
                    f"✅ Loaded from metadata: camera_position={camera_position}, surface_center={surface_center}"
                )
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not load metadata from {heatmap_metadata_path.name}: {e}"
                )

            return HeatmapResponse(
                success=True,
                output_path="",  # No PLY file in production mode
                message=f"Using existing heatmap metadata for query: '{request.query}'",
                target_position=target_position,
                surface_normal=list(surface_normal) if surface_normal else None,
                camera_position=list(camera_position) if camera_position else None,
                surface_center=list(surface_center) if surface_center else None,
            )
        elif metadata_exists and generate_ply and ply_exists:
            # Dev mode: Both metadata and PLY exist, use them
            print(f"✅ Using existing heatmap metadata: {heatmap_metadata_path.name}")
            print(f"✅ Using existing PLY file: {heatmap_ply_path.name}")

            # Load metadata from JSON file
            target_position = None
            surface_normal = None
            camera_position = None
            surface_center = None
            try:
                with open(heatmap_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    target_position = metadata.get("target_position")
                    surface_normal = metadata.get("surface_normal")
                    camera_position = metadata.get("camera_position")
                    surface_center = metadata.get("surface_center")
                print(
                    f"✅ Loaded from metadata: camera_position={camera_position}, surface_center={surface_center}"
                )
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not load metadata from {heatmap_metadata_path.name}: {e}"
                )

            # Return PLY file path
            try:
                relative_output = heatmap_ply_path.relative_to(source_root)
                http_path = f"/static/source/{relative_output.as_posix()}"
            except ValueError:
                relative_output = heatmap_ply_path.relative_to(project_root)
                http_path = f"/static/{relative_output.as_posix()}"

            return HeatmapResponse(
                success=True,
                output_path=http_path,
                message=f"Using existing heatmap files for query: '{request.query}'",
                target_position=target_position,
                surface_normal=list(surface_normal) if surface_normal else None,
                camera_position=list(camera_position) if camera_position else None,
                surface_center=list(surface_center) if surface_center else None,
            )
        elif metadata_exists and generate_ply and not ply_exists:
            # Dev mode: Metadata exists but PLY doesn't, generate PLY
            print(f"✅ Using existing heatmap metadata: {heatmap_metadata_path.name}")
            print(f"⚠️  PLY file not found, generating new PLY file...")

            # Load existing metadata
            target_position = None
            surface_normal = None
            camera_position = None
            surface_center = None
            try:
                with open(heatmap_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    target_position = metadata.get("target_position")
                    surface_normal = metadata.get("surface_normal")
                    camera_position = metadata.get("camera_position")
                    surface_center = metadata.get("surface_center")
                print(
                    f"✅ Loaded from metadata: camera_position={camera_position}, surface_center={surface_center}"
                )
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not load metadata from {heatmap_metadata_path.name}: {e}"
                )
            # Continue to PLY generation below (don't return here)
            # Use existing target_position if available, otherwise will be recalculated
            print(f"🔄 Generating new PLY file for existing metadata [DEV MODE]")
        elif not metadata_exists:
            # No metadata exists, generate from scratch
            if generate_ply:
                print(
                    f"🔄 Generating new heatmap (PLY + JSON) for query: '{request.query}' [DEV MODE]"
                )
            else:
                print(
                    f"🔄 Generating heatmap metadata (JSON only) for query: '{request.query}' [PRODUCTION MODE]"
                )

        try:
            # Check for preset first
            query_mode = os.getenv("QUERY_MODE", "sim_query")
            preset = None
            dataset_path = cf3_file.parent if request.vehicle_id else cf3_file.parent

            if query_mode in ["fixed_preset", "sim_query"]:
                preset = await asyncio.to_thread(
                    load_query_preset, str(dataset_path), request.query
                )

            # If preset found and we're in fixed_preset mode or don't need PLY
            if preset is not None:
                surface_normal, camera_position, surface_center = preset
                print(f"[Mode] Using preset from JSON (skipping computation)")

                # Check if heatmap PLY file already exists
                if generate_ply and heatmap_ply_path.exists():
                    print(
                        f"[Mode] Heatmap file already exists: {heatmap_ply_path.name}"
                    )

                    # Get relative path for HTTP access
                    try:
                        relative_output = heatmap_ply_path.relative_to(source_root)
                        http_path = f"/static/source/{relative_output.as_posix()}"
                    except ValueError:
                        relative_output = heatmap_ply_path.relative_to(project_root)
                        http_path = f"/static/{relative_output.as_posix()}"

                    return HeatmapResponse(
                        success=True,
                        output_path=http_path,
                        message=f"Using preset for query: '{request.query}'",
                        target_position=None,
                        surface_normal=list(surface_normal),
                        camera_position=list(camera_position),
                        surface_center=list(surface_center),
                    )
                elif not generate_ply:
                    # Production mode: no PLY needed, just return preset
                    return HeatmapResponse(
                        success=True,
                        output_path="",
                        message=f"Using preset for query: '{request.query}'",
                        target_position=None,
                        surface_normal=list(surface_normal),
                        camera_position=list(camera_position),
                        surface_center=list(surface_center),
                    )
                # else: generate PLY with skip_camera_calculation=True

            elif preset is None and query_mode == "fixed_preset":
                # Fixed preset mode but no preset found
                # 이전에는 404를 반환했지만, 이제는 프리셋이 없을 경우에도
                # on-the-fly 계산을 수행하도록 fallback 한다.
                print(
                    f"[Mode] QUERY_MODE=fixed_preset, but no preset found for "
                    f"query '{request.query}'. Falling back to dynamic heatmap generation."
                )

            # Generate PLY file and/or calculate target_position
            # Run in thread pool to avoid blocking event loop
            if generate_ply:
                # Development mode: Generate PLY file + metadata
                # Load vehicle center and mean vector if available
                vehicle_center = None
                mean_vector_path = None
                pose_file = dataset_path / "point_cloud_pruned_pose.json"
                if pose_file.exists():
                    try:
                        with open(pose_file, "r") as f:
                            pose_data = json.load(f)
                            if "center" in pose_data:
                                vehicle_center = tuple(pose_data["center"])
                    except:
                        pass

                # Always provide mean_vector_path so it can be auto-generated if missing
                mean_vector_file = dataset_path / "mean_vector.pt"
                mean_vector_path = str(mean_vector_file)

                # Note: Semantic centering is automatically enabled when mean_vector_path is provided
                # This matches spark's behavior

                skip_camera = preset is not None

                (
                    output_path,
                    calculated_target_position,
                    surface_normal,
                    camera_position,
                    surface_center,
                ) = await asyncio.to_thread(
                    generate_heatmap,
                    cf3_path=str(cf3_file),
                    reference_path=str(ref_file),
                    query=request.query,
                    ae_model_path=str(ae_file) if use_autoencoder else None,
                    output_dir=str(output_dir),
                    use_autoencoder=use_autoencoder,
                    vehicle_center=vehicle_center,
                    mean_vector_path=mean_vector_path,  # Auto-enables semantic centering
                    skip_camera_calculation=skip_camera,
                )

                # Use preset values if skip_camera was True
                if skip_camera and preset is not None:
                    surface_normal, camera_position, surface_center = preset

                # Save preset if new calculation was done
                if not skip_camera and surface_normal is not None:
                    await asyncio.to_thread(
                        save_query_preset,
                        str(dataset_path),
                        request.query,
                        surface_normal,
                        camera_position,
                        surface_center,
                    )

                target_position = calculated_target_position

                # Get relative path for HTTP access (PLY file)
                try:
                    relative_output = Path(output_path).relative_to(source_root)
                    http_path = f"/static/source/{relative_output.as_posix()}"
                except ValueError:
                    relative_output = Path(output_path).relative_to(project_root)
                    http_path = f"/static/{relative_output.as_posix()}"
            else:
                # Production mode: Only calculate target_position, no PLY generation
                target_position = await asyncio.to_thread(
                    generate_heatmap_target_position_only,
                    cf3_path=str(cf3_file),
                    query=request.query,
                    ae_model_path=str(ae_file) if use_autoencoder else None,
                    use_autoencoder=use_autoencoder,
                )
                output_path = None
                http_path = ""

                # Initialize camera-related variables to None in production mode
                # They may be set from preset if available
                if preset is not None:
                    surface_normal, camera_position, surface_center = preset
                else:
                    surface_normal = None
                    camera_position = None
                    surface_center = None

                # Note: In production mode without PLY, we don't calculate surface normal
                # This path should ideally use presets

            # Save metadata (target_position) for future use
            try:
                metadata = {
                    "query": request.query,
                    "vehicle_id": request.vehicle_id,
                    "use_autoencoder": use_autoencoder,
                    "target_position": (
                        list(target_position) if target_position else None
                    ),
                    "surface_normal": (
                        list(surface_normal) if surface_normal else None
                    ),
                    "camera_position": (
                        list(camera_position) if camera_position else None
                    ),
                    "surface_center": (
                        list(surface_center) if surface_center else None
                    ),
                }
                with open(heatmap_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"Metadata saved to: {heatmap_metadata_path}")
            except Exception as metadata_error:
                print(f"Warning: Failed to save metadata: {metadata_error}")

        except Exception as gen_error:
            import traceback

            error_trace = traceback.format_exc()
            print(f"ERROR in generate_heatmap: {str(gen_error)}")
            print(f"Traceback: {error_trace}")
            raise

        return HeatmapResponse(
            success=True,
            output_path=http_path,
            message=f"Heatmap metadata generated for query: '{request.query}'"
            + (" (PLY file also generated)" if generate_ply else " (JSON only)"),
            target_position=list(target_position) if target_position else None,
            surface_normal=list(surface_normal) if surface_normal else None,
            camera_position=list(camera_position) if camera_position else None,
            surface_center=list(surface_center) if surface_center else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"Error generating heatmap: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {str(e)}. Check server logs for details.",
        )
