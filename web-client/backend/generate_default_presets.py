#!/usr/bin/env python3
"""
Generate default camera presets for a vehicle dataset.

Creates 8 standard camera presets:
- initial: Initial overview of the vehicle (first load view)
- front: Front view
- back: Rear view  
- left: Left side view
- right: Right side view
- top: Top-down view
- front_windshield: Front windshield view (slightly elevated)
- rear_windshield: Rear windshield view (slightly elevated)

All presets use the vehicle center as the target point.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_vector(vec) -> np.ndarray:
    """Normalize a vector to unit length."""
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def calculate_camera_position(
    center: List[float],
    direction: np.ndarray,
    distance: float
) -> List[float]:
    """
    Calculate camera position from center along direction at given distance.
    
    Args:
        center: Target center point
        direction: Direction vector (will be normalized)
        distance: Distance from center
    
    Returns:
        Camera position [x, y, z]
    """
    direction_norm = normalize_vector(direction)
    position = np.array(center) + direction_norm * distance
    return position.tolist()


def generate_default_presets(
    center: List[float],
    forward: List[float],
    side: List[float],
    up: List[float],
    bbox_local: Dict[str, List[float]]
) -> Dict[str, Dict]:
    """
    Generate 8 default camera presets.
    
    Args:
        center: Vehicle center point [x, y, z]
        forward: Forward direction vector
        side: Side direction vector
        up: Up direction vector
        bbox_local: Bounding box with height, width, length ranges
    
    Returns:
        Dictionary of presets with camera_position and target
    """
    # Calculate optimal distance based on bbox
    height = bbox_local['height'][1] - bbox_local['height'][0]
    width = bbox_local['width'][1] - bbox_local['width'][0]
    length = bbox_local['length'][1] - bbox_local['length'][0]
    max_dimension = max(height, width, length)
    base_distance = max_dimension * 1.2
    
    print(f"📏 Vehicle dimensions: H={height:.2f}, W={width:.2f}, L={length:.2f}")
    print(f"📏 Base camera distance: {base_distance:.2f}")
    
    # Convert to numpy arrays
    forward_vec = np.array(forward)
    side_vec = np.array(side)
    up_vec = np.array(up)
    
    presets = {}
    
    # 0. INITIAL VIEW: Overview position (front-right-top)
    # Combine forward, side, and up directions for a nice overview angle
    initial_dir = normalize_vector(forward_vec * 1.0 + side_vec * -0.7 + up_vec * 0.6)
    initial_distance = base_distance * 1.4  # Slightly farther for overview
    initial_pos = calculate_camera_position(center, initial_dir, initial_distance)
    presets['initial'] = {
        'camera_position': initial_pos,
        'target': center,
        'description': 'Initial overview of vehicle (first load view)'
    }
    print(f"✅ Initial preset: camera at {[f'{x:.3f}' for x in initial_pos]}")
    
    # 1. FRONT VIEW: Camera in front of vehicle looking back
    # Position along forward direction
    front_pos = calculate_camera_position(center, forward_vec, base_distance)
    presets['front'] = {
        'camera_position': front_pos,
        'target': center,
        'description': 'Front view of vehicle'
    }
    print(f"✅ Front preset: camera at {[f'{x:.3f}' for x in front_pos]}")
    
    # 2. BACK VIEW: Camera behind vehicle looking forward
    # Position opposite to forward direction
    back_pos = calculate_camera_position(center, -forward_vec, base_distance)
    presets['back'] = {
        'camera_position': back_pos,
        'target': center,
        'description': 'Rear view of vehicle'
    }
    print(f"✅ Back preset: camera at {[f'{x:.3f}' for x in back_pos]}")
    
    # 3. LEFT VIEW: Camera on left side
    # Position along side direction
    left_pos = calculate_camera_position(center, side_vec, base_distance)
    presets['left'] = {
        'camera_position': left_pos,
        'target': center,
        'description': 'Left side view of vehicle'
    }
    print(f"✅ Left preset: camera at {[f'{x:.3f}' for x in left_pos]}")
    
    # 4. RIGHT VIEW: Camera on right side
    # Position opposite to side direction
    right_pos = calculate_camera_position(center, -side_vec, base_distance)
    presets['right'] = {
        'camera_position': right_pos,
        'target': center,
        'description': 'Right side view of vehicle'
    }
    print(f"✅ Right preset: camera at {[f'{x:.3f}' for x in right_pos]}")
    
    # 5. TOP VIEW: Camera above vehicle looking down
    # Position along up direction
    top_distance = base_distance * 1.2  # Slightly farther for top view
    top_pos = calculate_camera_position(center, up_vec, top_distance)
    presets['top'] = {
        'camera_position': top_pos,
        'target': center,
        'description': 'Top-down view of vehicle'
    }
    print(f"✅ Top preset: camera at {[f'{x:.3f}' for x in top_pos]}")
    
    # 6. FRONT WINDSHIELD: Front view elevated (30 degrees up)
    # Combine forward direction with upward tilt
    windshield_up_angle = 0.3  # Upward component (30% up)
    front_windshield_dir = normalize_vector(forward_vec + up_vec * windshield_up_angle)
    front_windshield_pos = calculate_camera_position(center, front_windshield_dir, base_distance)
    presets['front_windshield'] = {
        'camera_position': front_windshield_pos,
        'target': center,
        'description': 'Front windshield view (elevated)'
    }
    print(f"✅ Front windshield preset: camera at {[f'{x:.3f}' for x in front_windshield_pos]}")
    
    # 7. REAR WINDSHIELD: Rear view elevated (30 degrees up)
    # Combine backward direction with upward tilt
    rear_windshield_dir = normalize_vector(-forward_vec + up_vec * windshield_up_angle)
    rear_windshield_pos = calculate_camera_position(center, rear_windshield_dir, base_distance)
    presets['rear_windshield'] = {
        'camera_position': rear_windshield_pos,
        'target': center,
        'description': 'Rear windshield view (elevated)'
    }
    print(f"✅ Rear windshield preset: camera at {[f'{x:.3f}' for x in rear_windshield_pos]}")
    
    return presets


def save_presets_to_json(json_path: Path, presets: Dict[str, Dict]):
    """
    Save or update presets in the JSON file.
    
    Args:
        json_path: Path to point_cloud_pruned_pose.json
        presets: Dictionary of presets to add
    """
    # Load existing JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure camera_presets section exists
    if 'camera_presets' not in data:
        data['camera_presets'] = {}
    
    # Add/update presets
    for preset_name, preset_data in presets.items():
        data['camera_presets'][preset_name] = preset_data
        print(f"💾 Saved preset: {preset_name}")
    
    # Write back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Updated {json_path}")


def main(dataset_name: str = 'test'):
    """Generate and save default presets for a dataset."""
    # Paths
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / 'source' / dataset_name / 'point_cloud_pruned_pose.json'
    
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        return
    
    # Load JSON
    with open(json_path, 'r') as f:
        pose_data = json.load(f)
    
    print(f"\n📂 Processing dataset: {dataset_name}")
    print(f"📍 Center: {pose_data['center']}")
    
    # Check if camera_presets already exist
    if 'camera_presets' in pose_data:
        existing_presets = set(pose_data['camera_presets'].keys())
        default_preset_names = {
            'initial', 'front', 'back', 'left', 'right', 'top',
            'front_windshield', 'rear_windshield'
        }
        missing_presets = default_preset_names - existing_presets
        
        if not missing_presets:
            print("✅ All default presets already exist. Skipping generation.")
            return
        else:
            print(f"⚠️  Missing presets: {missing_presets}")
            print("📝 Generating missing presets...")
    
    # Generate presets
    presets = generate_default_presets(
        center=pose_data['center'],
        forward=pose_data['forward'],
        side=pose_data['side'],
        up=pose_data['up'],
        bbox_local=pose_data['bbox_local']
    )
    
    # Save to JSON
    save_presets_to_json(json_path, presets)
    
    print("\n" + "="*60)
    print("✅ Default camera presets generated successfully!")
    print("="*60)
    print("\nGenerated presets:")
    for preset_name in presets.keys():
        print(f"  - {preset_name}")


if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'test'
    main(dataset_name)
