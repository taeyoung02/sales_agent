"""
Camera presets router for Interactive AI Dealer backend
Handles camera preset generation and retrieval
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import subprocess
from utils.path_utils import get_source_root, get_source_file_path

router = APIRouter(prefix="/api/camera-presets", tags=["camera-presets"])


@router.post("/generate/{vehicle_id}")
async def generate_presets(vehicle_id: str):
    """
    Generate default camera presets for a vehicle dataset.
    
    This endpoint runs the generate_default_presets.py script to create
    standard camera views (front, back, left, right, top, windshields).
    
    Args:
        vehicle_id: ID of the vehicle dataset
        
    Returns:
        Success message with list of generated presets
    """
    try:
        # Get script path
        backend_dir = Path(__file__).parent.parent
        script_path = backend_dir / "generate_default_presets.py"
        
        if not script_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Preset generation script not found: {script_path}"
            )
        
        # Run the script
        print(f"📷 Generating camera presets for {vehicle_id}...")
        result = subprocess.run(
            ["python", str(script_path), vehicle_id],
            capture_output=True,
            text=True,
            cwd=str(backend_dir)
        )
        
        if result.returncode != 0:
            print(f"❌ Preset generation failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Preset generation failed: {result.stderr}"
            )
        
        print(f"✅ Presets generated for {vehicle_id}")
        print(result.stdout)
        
        # Load and return the generated presets
        source_root = get_source_root()
        json_path = source_root / vehicle_id / "point_cloud_pruned_pose.json"
        
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"JSON file not found: {json_path}"
            )
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'camera_presets' not in data:
            raise HTTPException(
                status_code=500,
                detail="Presets were generated but not found in JSON"
            )
        
        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "presets": list(data['camera_presets'].keys()),
            "message": f"Generated {len(data['camera_presets'])} camera presets"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating presets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate presets: {str(e)}"
        )


@router.get("/{vehicle_id}")
async def get_presets(vehicle_id: str):
    """
    Get all camera presets for a vehicle.
    Auto-generates default presets if they don't exist.
    
    Args:
        vehicle_id: ID of the vehicle dataset
        
    Returns:
        Dictionary of camera presets
    """
    try:
        source_root = get_source_root()
        json_path = source_root / vehicle_id / "point_cloud_pruned_pose.json"
        
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle data not found: {vehicle_id}"
            )
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Auto-generate presets if they don't exist
        if 'camera_presets' not in data or not data['camera_presets']:
            print(f"📷 No presets found for {vehicle_id}, auto-generating...")
            
            # Run the preset generation script
            backend_dir = Path(__file__).parent.parent
            script_path = backend_dir / "generate_default_presets.py"
            
            if script_path.exists():
                result = subprocess.run(
                    ["python", str(script_path), vehicle_id],
                    capture_output=True,
                    text=True,
                    cwd=str(backend_dir)
                )
                
                if result.returncode == 0:
                    print(f"✅ Auto-generated presets for {vehicle_id}")
                    # Reload the JSON to get the new presets
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                else:
                    print(f"⚠️  Failed to auto-generate presets: {result.stderr}")
        
        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "presets": data.get('camera_presets', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting presets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get presets: {str(e)}"
        )


@router.get("/{vehicle_id}/{preset_name}")
async def get_preset(vehicle_id: str, preset_name: str):
    """
    Get a specific camera preset for a vehicle.
    Auto-generates default presets if they don't exist.
    
    Args:
        vehicle_id: ID of the vehicle dataset
        preset_name: Name of the preset (e.g., 'front', 'top')
        
    Returns:
        Camera preset data
    """
    try:
        source_root = get_source_root()
        json_path = source_root / vehicle_id / "point_cloud_pruned_pose.json"
        
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle data not found: {vehicle_id}"
            )
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Auto-generate presets if they don't exist
        if 'camera_presets' not in data or preset_name not in data['camera_presets']:
            print(f"📷 Preset '{preset_name}' not found for {vehicle_id}, auto-generating...")
            
            # Run the preset generation script
            backend_dir = Path(__file__).parent.parent
            script_path = backend_dir / "generate_default_presets.py"
            
            if script_path.exists():
                result = subprocess.run(
                    ["python", str(script_path), vehicle_id],
                    capture_output=True,
                    text=True,
                    cwd=str(backend_dir)
                )
                
                if result.returncode == 0:
                    print(f"✅ Auto-generated presets for {vehicle_id}")
                    # Reload the JSON to get the new presets
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                else:
                    print(f"⚠️  Failed to auto-generate presets: {result.stderr}")
            
            # Check again after generation attempt
            if 'camera_presets' not in data or preset_name not in data['camera_presets']:
                raise HTTPException(
                    status_code=404,
                    detail=f"Preset '{preset_name}' not found for vehicle {vehicle_id}"
                )
        
        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "preset_name": preset_name,
            "preset": data['camera_presets'][preset_name]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting preset: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get preset: {str(e)}"
        )
