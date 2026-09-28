"""
Camera preset utilities for calculating and storing default camera views
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_camera_position(
    center: np.ndarray,
    axis: np.ndarray,
    distance: float,
    reverse: bool = False
) -> List[float]:
    """
    Calculate camera position from center and axis direction
    
    Args:
        center: Center point
        axis: Axis direction (forward/up/side)
        distance: Distance from center
        reverse: If True, camera is positioned opposite to axis direction
    
    Returns:
        Camera position as [x, y, z]
    """
    normalized = normalize_vector(axis)
    direction = -1 if reverse else 1
    
    position = center + normalized * distance * direction
    return position.tolist()


def calculate_optimal_distance(bbox: Optional[Dict[str, List[float]]]) -> float:
    """Calculate optimal camera distance based on bounding box"""
    if not bbox:
        return 5.0
    
    height = bbox['height'][1] - bbox['height'][0]
    width = bbox['width'][1] - bbox['width'][0]
    length = bbox['length'][1] - bbox['length'][0]
    
    max_dimension = max(height, width, length)
    distance = max_dimension * 1.5
    
    return distance


def calculate_all_camera_presets(pose_data: Dict) -> Dict[str, Dict]:
    """
    Calculate all default camera presets for a vehicle
    
    Args:
        pose_data: Camera pose data with center, forward, up, side, bbox_local
    
    Returns:
        Dictionary of preset names to camera configurations (in JSON coordinate system)
    """
    # Keep JSON coordinates as-is (frontend will handle PLY transformation when rendering)
    center = np.array(pose_data['center'])
    forward = np.array(pose_data['forward'])
    up = np.array(pose_data['up'])
    side = np.array(pose_data['side'])
    
    # FIX: JSON의 forward와 side가 바뀌어 있음
    # 실제로 forward가 side이고, side가 forward임
    # 따라서 서로 바꿔서 사용
    actual_forward = side  # JSON의 side가 실제 차량의 forward
    actual_side = forward  # JSON의 forward가 실제 차량의 side
    
    bbox = pose_data.get('bbox_local')
    
    # Calculate optimal distance
    distance = calculate_optimal_distance(bbox)
    
    print(f"📊 Using axes from JSON (with swap fix):")
    print(f"  Center: {center}")
    print(f"  JSON Forward (실제 Side): {forward}")
    print(f"  JSON Side (실제 Forward): {side}")
    print(f"  → Actual Forward: {actual_forward}")
    print(f"  → Actual Side: {actual_side}")
    print(f"  Up: {up}")
    print(f"  Distance: {distance:.2f}")
    
    presets = {}
    
    # Front view: Camera positioned along actual_forward direction, looking at center
    front_pos = np.array(calculate_camera_position(center, actual_forward, distance, False))
    presets['front'] = {
        'position': front_pos.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Front: position={front_pos}, target={center}")
    
    # Back view: Camera positioned opposite to actual_forward direction, looking at center
    back_pos = np.array(calculate_camera_position(center, actual_forward, distance, True))
    presets['back'] = {
        'position': back_pos.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Back: position={back_pos}, target={center}")
    
    # Left view: Camera positioned along actual_side direction (left of vehicle)
    left_pos = np.array(calculate_camera_position(center, actual_side, distance, False))
    presets['left'] = {
        'position': left_pos.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Left: position={left_pos}, target={center}")
    
    # Right view: Camera positioned opposite to actual_side direction (right of vehicle)
    right_pos = np.array(calculate_camera_position(center, actual_side, distance, True))
    presets['right'] = {
        'position': right_pos.tolist(),
        'target': center.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Right: position={right_pos}, target={center}")
    
    # Top view: Camera positioned along up direction, looking down
    top_pos = np.array(calculate_camera_position(center, up, distance, False))
    presets['top'] = {
        'position': top_pos.tolist(),
        'target': center.tolist(),
        'up': actual_forward.tolist(),  # When looking from top, forward is up
    }
    print(f"  📷 Top: position={top_pos}, target={center}, up={actual_forward}")
    
    # Front windshield: Front view but elevated (looking down at windshield)
    # Move camera up along the up vector
    windshield_elevation = distance * 0.3  # Elevate by 30% of distance
    front_windshield_pos = front_pos + up * windshield_elevation
    
    # Target is slightly above center (windshield height)
    windshield_target = center.copy()
    if bbox:
        # Target the upper part of the vehicle
        bbox_height_max = bbox['height'][1]
        windshield_target[1] = bbox_height_max * 0.7  # 70% up from ground
    
    presets['front-windshield'] = {
        'position': front_windshield_pos.tolist(),
        'target': windshield_target.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Front-windshield: position={front_windshield_pos}, target={windshield_target}")
    
    # Rear windshield: Back view but elevated (looking down at rear windshield)
    rear_windshield_pos = back_pos + up * windshield_elevation
    
    presets['rear-windshield'] = {
        'position': rear_windshield_pos.tolist(),
        'target': windshield_target.tolist(),
        'up': up.tolist(),
    }
    print(f"  📷 Rear-windshield: position={rear_windshield_pos}, target={windshield_target}")
    
    print(f"✅ Calculated {len(presets)} camera presets: {list(presets.keys())}")
    
    return presets


def ensure_camera_presets(dataset_path: str) -> bool:
    """
    Ensure camera presets exist in the pose JSON file
    If they don't exist, calculate and save them
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        True if presets were created, False if they already existed
    """
    pose_file = Path(dataset_path) / "point_cloud_pruned_pose.json"
    
    if not pose_file.exists():
        print(f"❌ Pose file not found: {pose_file}")
        return False
    
    # Load existing pose data
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    
    # Check if presets already exist
    if 'camera_presets' in pose_data and pose_data['camera_presets']:
        required_presets = {'front', 'back', 'left', 'right', 'top', 'front-windshield', 'rear-windshield'}
        existing_presets = set(pose_data['camera_presets'].keys())
        
        if required_presets.issubset(existing_presets):
            print(f"✅ Camera presets already exist in {pose_file.name}")
            return False
    
    # Calculate and save presets
    print(f"📷 Calculating camera presets for {dataset_path}...")
    presets = calculate_all_camera_presets(pose_data)
    
    pose_data['camera_presets'] = presets
    
    # Save updated pose data
    with open(pose_file, 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    print(f"💾 Saved camera presets to {pose_file.name}")
    
    return True


def get_camera_preset(dataset_path: str, preset_name: str) -> Optional[Dict]:
    """
    Get a specific camera preset from the pose JSON file
    
    Args:
        dataset_path: Path to dataset directory
        preset_name: Name of the preset (e.g., 'front', 'top')
    
    Returns:
        Camera preset data or None if not found
    """
    pose_file = Path(dataset_path) / "point_cloud_pruned_pose.json"
    
    if not pose_file.exists():
        return None
    
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    
    if 'camera_presets' not in pose_data:
        return None
    
    return pose_data['camera_presets'].get(preset_name)
