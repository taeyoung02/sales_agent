import json
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import torch

def create_gaussian_line(start, vector, length, color, num_points=100):
    """
    한 지점에서 벡터 방향으로 점들을 생성하여 선을 만듦
    color: [R, G, B] (0~1)
    """
    t = np.linspace(0, length, num_points)
    # (N, 3) 좌표 생성
    points = start + vector * t[:, np.newaxis]
    
    # 색상 (SH DC component로 변환 필요)
    # SH DC = (RGB - 0.5) / 0.28209479177387814
    C0 = 0.28209479177387814
    sh_colors = (color - 0.5) / C0
    colors = np.tile(sh_colors, (num_points, 1))
    
    return points, colors

def save_debug_ply(json_path, output_path, axis_length=2.0):
    # 1. JSON 로드
    with open(json_path, 'r') as f:
        pose = json.load(f)
    
    center = np.array(pose['center'])
    forward = np.array(pose['forward'])
    up = np.array(pose['up'])
    side = np.array(pose['side'])
    
    print(f"Loading Pose from {json_path}")
    print(f"Center: {center}")
    
    # 2. 가상 포인트 생성 (축 그리기)
    # 3DGS 포맷에 맞게 점들을 생성합니다.
    
    points_list = []
    colors_list = []
    
    # Center (Yellow Blob)
    points_list.append(center.reshape(1, 3))
    colors_list.append(np.array([[1.0, 1.0, 0.0]]) / 0.28209479177387814 - 0.5) # Yellow
    
    # Forward Axis (Blue)
    p, c = create_gaussian_line(center, forward, axis_length, np.array([0, 0, 1]))
    points_list.append(p)
    colors_list.append(c)
    
    # Up Axis (Green)
    p, c = create_gaussian_line(center, up, axis_length/2, np.array([0, 1, 0])) # 높이는 좀 짧게
    points_list.append(p)
    colors_list.append(c)
    
    # Side Axis (Red)
    p, c = create_gaussian_line(center, side, axis_length/1.5, np.array([1, 0, 0]))
    points_list.append(p)
    colors_list.append(c)
    
    # 데이터 합치기
    xyz = np.vstack(points_list)
    f_dc = np.vstack(colors_list) # features_dc (Color)
    
    num_points = xyz.shape[0]
    
    # 3. 3DGS PLY 속성 채우기
    # 필수 속성: x, y, z, opacity, scale_0~2, rot_0~3, f_dc_0~2
    
    # Opacity: 완전 불투명 (inverse sigmoid)
    # sigmoid(10) ~= 1.0
    opacities = np.ones(num_points) * 10.0 
    
    # Scale: 아주 작은 구체 (log scale)
    # exp(-5) ~= 0.006
    scales = np.ones((num_points, 3)) * -5.0
    # 중심점은 좀 크게
    scales[0] = -3.0 
    
    # Rotation: Identity (0, 0, 0, 1)
    rots = np.zeros((num_points, 4))
    rots[:, 0] = 1.0
    
    # PLY Element 생성
    # 구조체 정의
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), # Normals (dummy)
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    elements = np.zeros(num_points, dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rots[:, 0]
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Debug PLY saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Path to vehicle_pose.json')
    parser.add_argument('--out', default='debug_axes.ply', help='Output PLY path')
    parser.add_argument('--len', type=float, default=3.0, help='Length of the axis lines')
    
    args = parser.parse_args()
    save_debug_ply(args.json, args.out, args.len)