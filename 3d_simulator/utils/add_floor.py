import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import json
import os
from scipy.spatial.transform import Rotation

def load_pose_data(json_path):
    """mahalanobis_pruning.py에서 생성한 pose 정보 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_rotation_matrix_from_vectors(forward, side, up):
    """벡터로부터 회전 행렬 생성 (Right-Handed 보장 및 Determinant 에러 방지)"""
    # 1. 벡터 정규화
    forward = forward / np.linalg.norm(forward)
    side = side / np.linalg.norm(side)
    up = up / np.linalg.norm(up)

    # 2. 행렬 구성
    R = np.array([forward, side, up]).T

    # 3. Determinant 확인 (Left-handed 좌표계 보정)
    if np.linalg.det(R) < 0:
        side = -side
        R = np.array([forward, side, up]).T
        
    return R

def calculate_tight_floor_level(ply_path, pose_data):
    """
    Pruned PLY 파일을 분석하여 실제 가장 낮은 지점(Tight Floor)을 계산
    """
    print("Calculating tight floor level from point cloud...")
    
    # PLY 읽기
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # 점 데이터 (World Space)
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    # Pose 정보
    center = np.array(pose_data['center'])
    up = np.array(pose_data['up'])
    up = up / np.linalg.norm(up) # 정규화
    
    # 벡터 투영 (Project points onto Up vector)
    # 높이 h = (P - Center) dot Up
    vecs = pts - center
    heights = np.dot(vecs, up)
    
    # 하위 0.5% 지점을 바닥으로 설정 (노이즈 제외하고 가장 타이트한 지점)
    # 이미 Pruning된 데이터이므로 노이즈가 적어서 0.1~0.5%면 안전함
    floor_level_local = np.percentile(heights, 0.5)
    
    print(f"  - JSON bbox bottom: {pose_data['bbox_local']['height'][0]:.4f}")
    print(f"  - Recalculated tight bottom (0.5%): {floor_level_local:.4f}")
    
    return floor_level_local
def generate_floor_gaussians(pose_data, tight_floor_level=None, num_points=100000, color=(0.8, 0.8, 0.8), expansion_ratio=1.5):
    """
    JSON 정보를 바탕으로 바닥 Gaussian 생성 (High Density, Small Scale)
    """
    # 1. 데이터 추출
    center = np.array(pose_data['center'])
    forward = np.array(pose_data['forward'])
    side = np.array(pose_data['side'])
    up = np.array(pose_data['up'])
    
    # 로컬 바운딩 박스 정보
    bbox = pose_data['bbox_local']
    
    # 높이 결정
    if tight_floor_level is not None:
        floor_z = tight_floor_level
    else:
        floor_z = bbox['height'][0]
        
    width_radius = (bbox['width'][1] - bbox['width'][0]) / 2 * expansion_ratio
    length_radius = (bbox['length'][1] - bbox['length'][0]) / 2 * expansion_ratio
    
    # =========================================================================
    # [핵심 수정 1] 점 개수 증가 (num_points)
    # 인자로 받은 num_points를 사용 (기본값을 100,000개 정도로 늘리는 것 추천)
    # =========================================================================
    
    # 2. 로컬 좌표계에서 타원형 점 생성
    radii = np.sqrt(np.random.rand(num_points))
    thetas = np.random.rand(num_points) * 2 * np.pi
    
    local_x = radii * np.cos(thetas) * length_radius
    local_y = radii * np.sin(thetas) * width_radius
    
    # =========================================================================
    # [핵심 수정 2] Z Offset 조정
    # Splat이 위로 튀어 올라오는 것을 방지하기 위해 살짝 더 내림 (-0.03 ~ -0.05)
    # =========================================================================
    local_z = np.full(num_points, floor_z - 0.04) 
    
    local_points = np.stack([local_x, local_y, local_z], axis=1)
    
    # 3. 월드 좌표계로 변환
    R = get_rotation_matrix_from_vectors(forward, side, up)
    world_points = (R @ local_points.T).T + center
    
    # 4. 3DGS 속성 생성
    
    # =========================================================================
    # [핵심 수정 3] Scale 대폭 축소 (Flat & Small)
    # 기존: np.log(0.1 + ...) -> 약 0.1 크기 (너무 큼)
    # 수정: np.log(0.015 + ...) -> 약 0.015 ~ 0.025 크기 (기존 대비 1/5 수준)
    # =========================================================================
    
    # XY Scale: 아주 작게 설정하여 '모래'처럼 만듦
    scale_xy_val = 0.015 + np.random.rand(num_points) * 0.015
    scale_xy = np.log(scale_xy_val) 
    
    # Z Scale: 바닥 두께는 종이장처럼 얇게
    scale_z_val = 0.0005
    scale_z = np.full(num_points, np.log(scale_z_val))
    
    scales = np.stack([scale_xy, scale_xy, scale_z], axis=1)
    
    # Rotation (Quaternion) - 바닥에 평행하게 눕힘
    r = Rotation.from_matrix(R)
    quats = r.as_quat() # (x, y, z, w)
    quats = np.tile(np.array([quats[3], quats[0], quats[1], quats[2]]), (num_points, 1))
    
    # Opacity (Solid)
    # 너무 투명하면 뒤의 노이즈가 보일 수 있으니 불투명하게 하되, 
    # 점이 작아서 겹쳐지면서 자연스러워짐
    opacities = np.full((num_points, 1), 3.0) # sigmoid(3.0) ~= 0.95
    
    # Color (SH DC)
    SH_C0 = 0.28209479177387814
    sh_val = (np.array(color) - 0.5) / SH_C0
    f_dc = np.tile(sh_val, (num_points, 1))
    
    return {
        'xyz': world_points,
        'scale': scales,
        'rot': quats,
        'opacity': opacities,
        'f_dc': f_dc
    }
    
def merge_and_save(original_ply_path, floor_data, output_path):
    """원본 PLY와 바닥 데이터를 합쳐서 저장"""
    print(f"Reading original PLY to merge: {original_ply_path}")
    plydata = PlyData.read(original_ply_path)
    orig_vertex = plydata['vertex']
    
    num_orig = len(orig_vertex['x'])
    num_floor = len(floor_data['xyz'])
    total_points = num_orig + num_floor
    
    # 새 데이터 배열
    new_dtype = orig_vertex.data.dtype
    new_data = np.zeros(total_points, dtype=new_dtype)
    
    # 1. 원본 복사
    for name in new_dtype.names:
        new_data[name][:num_orig] = orig_vertex[name]
        
    # 2. 바닥 데이터 채우기
    new_data['x'][num_orig:] = floor_data['xyz'][:, 0]
    new_data['y'][num_orig:] = floor_data['xyz'][:, 1]
    new_data['z'][num_orig:] = floor_data['xyz'][:, 2]
    
    new_data['opacity'][num_orig:] = floor_data['opacity'][:, 0]
    
    if 'scale_0' in new_dtype.names:
        new_data['scale_0'][num_orig:] = floor_data['scale'][:, 0]
        new_data['scale_1'][num_orig:] = floor_data['scale'][:, 1]
        new_data['scale_2'][num_orig:] = floor_data['scale'][:, 2]
        
    if 'rot_0' in new_dtype.names:
        new_data['rot_0'][num_orig:] = floor_data['rot'][:, 0]
        new_data['rot_1'][num_orig:] = floor_data['rot'][:, 1]
        new_data['rot_2'][num_orig:] = floor_data['rot'][:, 2]
        new_data['rot_3'][num_orig:] = floor_data['rot'][:, 3]
        
    if 'f_dc_0' in new_dtype.names:
        new_data['f_dc_0'][num_orig:] = floor_data['f_dc'][:, 0]
        new_data['f_dc_1'][num_orig:] = floor_data['f_dc'][:, 1]
        new_data['f_dc_2'][num_orig:] = floor_data['f_dc'][:, 2]
        
    el = PlyElement.describe(new_data, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved merged PLY to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input Pruned PLY file')
    parser.add_argument('-j', '--json', required=True, help='Input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output PLY file')
    parser.add_argument('--points', type=int, default=30000, help='Number of points for the floor')
    parser.add_argument('--color', type=float, nargs=3, default=[0.05, 0.05, 0.05], help='Floor color RGB')
    parser.add_argument('--expand', type=float, default=1.3, help='Expansion ratio')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input PLY not found: {args.input}")
        exit(1)
        
    # 1. JSON 로드
    pose_info = load_pose_data(args.json)
    
    # 2. [핵심] PLY 파일을 직접 읽어서 Tight한 바닥 높이 계산
    tight_floor_level = calculate_tight_floor_level(args.input, pose_info)
    
    # 3. 바닥 생성 (재계산된 높이 사용)
    floor_data = generate_floor_gaussians(
        pose_info,
        tight_floor_level=tight_floor_level,
        num_points=args.points, 
        color=tuple(args.color), 
        expansion_ratio=args.expand
    )
    
    # 4. 병합 및 저장
    merge_and_save(args.input, floor_data, args.output)