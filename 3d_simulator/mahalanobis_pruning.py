import numpy as np
from plyfile import PlyData, PlyElement
import os
import argparse
import torch
import json
import struct
from collections import namedtuple

# ============================================================================
# COLMAP Data Reading Functions
# ============================================================================

def qvec2rotmat(qvec):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        qvec: Quaternion [qw, qx, qy, qz]
    
    Returns:
        3x3 rotation matrix
    """
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2,
         2 * x * y - 2 * w * z,
         2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z,
         1 - 2 * x**2 - 2 * z**2,
         2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y,
         2 * y * z + 2 * w * x,
         1 - 2 * x**2 - 2 * y**2]
    ])

def read_colmap_images_binary(images_bin_path):
    """
    Read COLMAP images.bin file.
    
    Returns:
        dict: {image_id: {'qvec': array, 'tvec': array, 'name': str}}
    """
    images = {}
    
    with open(images_bin_path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_images):
            # Read image entry
            image_id = struct.unpack('I', f.read(4))[0]
            qvec = struct.unpack('dddd', f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack('ddd', f.read(24))    # tx, ty, tz
            camera_id = struct.unpack('I', f.read(4))[0]
            
            # Read image name
            name_bytes = b''
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_bytes += char
            name = name_bytes.decode('utf-8')
            
            # Skip 2D points
            num_points2D = struct.unpack('Q', f.read(8))[0]
            f.read(24 * num_points2D)  # Skip x, y, point3D_id for each point
            
            images[image_id] = {
                'qvec': np.array(qvec),
                'tvec': np.array(tvec),
                'name': name,
                'camera_id': camera_id
            }
    
    return images

def read_colmap_images_text(images_txt_path):
    """
    Read COLMAP images.txt file.
    
    Returns:
        dict: {image_id: {'qvec': array, 'tvec': array, 'name': str}}
    """
    images = {}
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), 
                            float(parts[3]), float(parts[4])])
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            camera_id = int(parts[8])
            name = parts[9]
            
            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'name': name,
                'camera_id': camera_id
            }
    
    return images

def load_colmap_images(colmap_path):
    """
    Load COLMAP images from binary or text format.
    
    Args:
        colmap_path: Path to COLMAP sparse folder (containing images.bin or images.txt)
    
    Returns:
        dict: COLMAP images data
    """
    images_bin = os.path.join(colmap_path, 'images.bin')
    images_txt = os.path.join(colmap_path, 'images.txt')
    
    if os.path.exists(images_bin):
        print(f"Loading COLMAP images from: {images_bin}")
        return read_colmap_images_binary(images_bin)
    elif os.path.exists(images_txt):
        print(f"Loading COLMAP images from: {images_txt}")
        return read_colmap_images_text(images_txt)
    else:
        print(f"Warning: No COLMAP images file found in {colmap_path}")
        return None

# ============================================================================
# Axis Alignment Functions
# ============================================================================

def align_axes_hemisphere(pose_info, colmap_images):
    """
    Align Up/Down axis using camera hemisphere heuristic.
    
    Logic: Most cameras are positioned ABOVE the vehicle (not underground).
    Therefore, vectors from vehicle center to cameras should point upward.
    
    Args:
        pose_info: dict with 'center', 'up', 'forward', 'side' vectors
        colmap_images: dict from load_colmap_images()
    
    Returns:
        Updated pose_info with corrected up vector
    """
    if colmap_images is None or len(colmap_images) == 0:
        print("[Axis Alignment] No COLMAP data available, skipping hemisphere check")
        return pose_info
    
    center = np.array(pose_info['center'])
    up = np.array(pose_info['up'])
    
    print("\n[Axis Alignment] Verifying Up/Down axis direction...")
    
    # Extract camera positions in world coordinates
    cam_centers = []
    for img_id, img_data in colmap_images.items():
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # COLMAP convention: R * X + t = 0  =>  X = -R^T * t
        R = qvec2rotmat(qvec)
        cam_pos = -R.T @ tvec
        cam_centers.append(cam_pos)
    
    cam_centers = np.array(cam_centers)
    
    # Vectors from vehicle center to cameras
    vecs_to_camera = cam_centers - center
    
    # Project onto current up vector
    dot_products = vecs_to_camera @ up
    mean_dot = np.mean(dot_products)
    
    print(f"  - Mean dot product (camera_vec · up_vec): {mean_dot:.6f}")
    print(f"  - Camera positions above center: {np.sum(dot_products > 0)}/{len(dot_products)}")
    
    # If mean dot product is negative, cameras are below the center
    # This means up vector is pointing down -> flip it
    if mean_dot < 0:
        print("  ✓ CORRECTION: Up axis was pointing downwards. FLIPPING it.")
        up = -up
        pose_info['up'] = up.tolist()
    else:
        print("  ✓ Up axis is correct (pointing toward camera hemisphere)")
    
    return pose_info

def align_axes_clip(pose_info, colmap_images, image_folder, device='cuda'):
    """
    Align Forward/Backward axis using CLIP verification.
    
    Logic: Use CLIP to determine if the "forward" direction shows front or rear.
    
    Args:
        pose_info: dict with 'center', 'forward', 'up', 'side' vectors
        colmap_images: dict from load_colmap_images()
        image_folder: Path to folder containing images
        device: 'cuda' or 'cpu'
    
    Returns:
        Updated pose_info with corrected forward vector
    """
    if colmap_images is None or len(colmap_images) == 0:
        print("[Axis Alignment] No COLMAP data available, skipping CLIP check")
        return pose_info
    
    # Try to import CLIP
    try:
        import clip
        from PIL import Image as PILImage
    except ImportError:
        print("[Axis Alignment] CLIP not available (pip install git+https://github.com/openai/CLIP.git)")
        print("  Skipping forward/backward axis verification")
        return pose_info
    
    print("\n[Axis Alignment] Verifying Forward/Backward axis direction...")
    
    center = np.array(pose_info['center'])
    forward = np.array(pose_info['forward'])
    forward = forward / np.linalg.norm(forward)  # Normalize
    
    # Load CLIP model
    try:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"  - CLIP model loaded on {device}")
    except Exception as e:
        print(f"  Warning: Failed to load CLIP model: {e}")
        return pose_info
    
    # Calculate view direction for each camera
    cam_view_dots = []
    cam_data = []
    
    for img_id, img_data in colmap_images.items():
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # Camera position in world coordinates
        R = qvec2rotmat(qvec)
        cam_pos = -R.T @ tvec
        
        # Vector from vehicle to camera (view direction)
        view_vec = cam_pos - center
        view_vec = view_vec / np.linalg.norm(view_vec)
        
        # Dot product with forward axis
        dot = np.dot(view_vec, forward)
        
        cam_view_dots.append(dot)
        cam_data.append({
            'id': img_id,
            'name': img_data['name'],
            'dot': dot
        })
    
    # Find images most aligned with forward (+1) and backward (-1)
    cam_view_dots = np.array(cam_view_dots)
    idx_forward = np.argmax(cam_view_dots)  # Most aligned with +forward
    idx_backward = np.argmin(cam_view_dots)  # Most aligned with -forward
    
    img_forward_name = cam_data[idx_forward]['name']
    img_backward_name = cam_data[idx_backward]['name']
    
    print(f"  - Testing forward direction: {img_forward_name} (dot={cam_data[idx_forward]['dot']:.3f})")
    print(f"  - Testing backward direction: {img_backward_name} (dot={cam_data[idx_backward]['dot']:.3f})")
    
    # Load and preprocess image
    img_path = os.path.join(image_folder, img_forward_name)
    if not os.path.exists(img_path):
        print(f"  Warning: Image not found: {img_path}")
        return pose_info
    
    try:
        image = preprocess(PILImage.open(img_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"  Warning: Failed to load image: {e}")
        return pose_info
    
    # CLIP text prompts
    text_prompts = [
        "a photo of the front view of a car",
        "a photo of the rear view of a car",
        "a photo of the back view of a vehicle"
    ]
    text_inputs = clip.tokenize(text_prompts).to(device)
    
    # Calculate similarities
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_inputs)
        
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    front_score = similarity[0, 0].item()
    rear_score_1 = similarity[0, 1].item()
    rear_score_2 = similarity[0, 2].item()
    rear_score = max(rear_score_1, rear_score_2)
    
    print(f"  - CLIP scores for '{img_forward_name}':")
    print(f"      Front view: {front_score:.3f}")
    print(f"      Rear view:  {rear_score:.3f}")
    
    # If rear score is higher, forward axis is actually pointing backward
    if rear_score > front_score:
        print("  ✓ CORRECTION: Forward axis is pointing to REAR. FLIPPING it.")
        forward = -forward
        pose_info['forward'] = forward.tolist()
        
        # Recalculate side vector (maintain right-hand rule: side = up × forward)
        up = np.array(pose_info['up'])
        side = np.cross(up, forward)
        pose_info['side'] = side.tolist()
    else:
        print("  ✓ Forward axis is correct (pointing to front)")
    
    return pose_info

def align_axes(pose_info, colmap_path=None, image_folder=None, device='cuda'):
    """
    Comprehensive axis alignment using multiple strategies.
    
    Args:
        pose_info: dict with vehicle pose information
        colmap_path: Path to COLMAP sparse folder (for images.bin/txt)
        image_folder: Path to folder containing images (for CLIP)
        device: 'cuda' or 'cpu'
    
    Returns:
        Updated pose_info with corrected axis directions
    """
    print("\n" + "="*60)
    print("Axis Alignment Verification")
    print("="*60)
    
    # Load COLMAP data if path provided
    colmap_images = None
    if colmap_path and os.path.exists(colmap_path):
        colmap_images = load_colmap_images(colmap_path)
    
    # Strategy 1: Hemisphere check (Up/Down)
    pose_info = align_axes_hemisphere(pose_info, colmap_images)
    
    # Strategy 2: CLIP verification (Forward/Backward)
    if image_folder and os.path.exists(image_folder):
        pose_info = align_axes_clip(pose_info, colmap_images, image_folder, device)
    else:
        print("\n[Axis Alignment] Image folder not provided, skipping CLIP verification")
    
    print("="*60 + "\n")
    
    return pose_info

def load_ply(path, device='cpu'):
    """PLY 파일을 읽어 PyTorch Tensor로 변환"""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # 3DGS PLY 속성 추출
    xyz = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1)
    opacity = np.array(vertex['opacity'])  # Logit space
    
    # Scale은 보통 scale_0, scale_1, scale_2로 저장됨 (Log space)
    scale_names = [n.name for n in vertex.properties if n.name.startswith('scale_')]
    scales = np.stack([vertex[n] for n in scale_names], axis=-1)
    
    # Tensor 변환
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    opa_t = torch.tensor(opacity, dtype=torch.float32, device=device)
    scl_t = torch.tensor(scales, dtype=torch.float32, device=device)
    
    return xyz_t, opa_t, scl_t, vertex

def save_ply(original_vertex, mask, output_path):
    """마스킹된 데이터를 다시 PLY로 저장"""
    # Tensor인 경우 numpy로 변환
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # 마스크된 데이터만 추출
    new_data = {}
    for prop in original_vertex.properties:
        new_data[prop.name] = original_vertex[prop.name][mask]
    
    # PlyElement 생성
    types = [(p.name, original_vertex[p.name].dtype) for p in original_vertex.properties]
    
    # numpy structured array 생성
    new_vertex_data = np.zeros(np.sum(mask), dtype=types)
    for name, dtype in types:
        new_vertex_data[name] = new_data[name]
        
    el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved pruned PLY to: {output_path}")

def save_metadata(center, axes, eigenvalues, output_path):
    """중심점과 주축 정보를 텍스트 파일로 저장
    
    Args:
        center: 3D 중심점 좌표 (x, y, z)
        axes: 3x3 주축 벡터 행렬 (eigenvectors)
        eigenvalues: 각 주축의 eigenvalue (분산)
        output_path: 저장할 txt 파일 경로
    """
    # eigenvalue 기준으로 정렬 (큰 순서대로)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_axes = axes[:, sorted_indices]
    
    with open(output_path, 'w') as f:
        f.write("=== Vehicle Metadata ===\n\n")
        
        f.write(f"Center: {center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}\n\n")
        
        f.write("Principal Axes (sorted by variance):\n")
        f.write(f"Primary Axis (longest):   [{sorted_axes[0,0]:8.5f}, {sorted_axes[1,0]:8.5f}, {sorted_axes[2,0]:8.5f}] (eigenvalue: {sorted_eigenvalues[0]:.6f})\n")
        f.write(f"Secondary Axis (medium):  [{sorted_axes[0,1]:8.5f}, {sorted_axes[1,1]:8.5f}, {sorted_axes[2,1]:8.5f}] (eigenvalue: {sorted_eigenvalues[1]:.6f})\n")
        f.write(f"Tertiary Axis (shortest): [{sorted_axes[0,2]:8.5f}, {sorted_axes[1,2]:8.5f}, {sorted_axes[2,2]:8.5f}] (eigenvalue: {sorted_eigenvalues[2]:.6f})\n\n")
        
        # Z축 방향 (보통 차량의 높이 방향은 가장 작은 분산)
        f.write("Note: The tertiary axis (shortest) typically represents the Z-axis (height direction)\n")
    
    print(f"Saved metadata to: {output_path}")

def auto_prune_vehicle(input_path, output_path, sigma_thresh=None, save_meta=True, use_auto_bounds=True, device=None, iterations=1, colmap_path=None, image_folder=None, convergence_threshold=0.01, min_score_percentile=0.7):
    """
    CUDA-accelerated vehicle pruning with automatic or manual threshold
    
    Args:
        input_path: Input PLY file path
        output_path: Output PLY file path  
        sigma_thresh: Manual sigma threshold (used only if use_auto_bounds=False)
        save_meta: Whether to save metadata
        use_auto_bounds: Use automatic histogram-based bounds detection
        device: 'cuda', 'cpu', or None (auto-detect)
        iterations: Number of times to apply pruning iteratively (default: 1)
        colmap_path: Path to COLMAP sparse folder for axis alignment verification
        image_folder: Path to images folder for CLIP-based axis alignment
        convergence_threshold: Stop if pruning ratio < this value (default: 0.01 = 1%)
        min_score_percentile: Minimum score percentile for core points (default: 0.7 = top 30%)
    """
    # Device 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Processing: {input_path}")
    print(f"Running on {device}...")
    print(f"Max Iterations: {iterations}")
    print(f"Convergence Threshold: {convergence_threshold * 100:.1f}% pruning rate")
    print(f"Core Score Percentile: Top {(1 - min_score_percentile) * 100:.0f}%\n")
    
    # Iterative pruning loop
    current_pts = None
    current_opa = None
    current_scl = None
    current_vertex = None
    prev_count = None
    
    for iter_num in range(1, iterations + 1):
        if iterations > 1:
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}/{iterations}")
            print(f"{'='*60}")
        
        # 1. Load Data (첫 iteration은 파일에서, 이후는 메모리에서)
        if iter_num == 1:
            pts, raw_opa, raw_scl, raw_vertex = load_ply(input_path, device)
        else:
            # 이전 iteration의 결과를 사용
            pts = current_pts
            raw_opa = current_opa
            raw_scl = current_scl
            raw_vertex = current_vertex
        
        # 2. Activation Function 적용 (Log space -> Linear space)
        opacities_act = torch.sigmoid(raw_opa)
        scales_act = torch.exp(raw_scl)
        
        # 3. Score 계산 (Objectness)
        # 작고(Scale이 작음) + 진한(Opacity가 큼) 점이 높은 점수
        mean_scale = torch.mean(scales_act, dim=1)
        epsilon = 1e-6
        scores = opacities_act / (mean_scale + epsilon)
        
        # 점수 정규화 (0~1)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # 4. Robust Center & PCA 계산 - 동적 threshold 사용
        # 첫 iteration: 상위 30% (보수적), 이후: min_score_percentile 사용
        if iter_num == 1:
            threshold_score = torch.quantile(scores, 0.7)  # 상위 30%
            print(f"Initial core selection: Top 30% (score > {threshold_score:.4f})")
        else:
            threshold_score = torch.quantile(scores, min_score_percentile)
            print(f"Refined core selection: Top {(1 - min_score_percentile) * 100:.0f}% (score > {threshold_score:.4f})")
        
        high_score_mask = scores > threshold_score
        
        core_pts = pts[high_score_mask]
        core_scores = scores[high_score_mask]
        
        print(f"Core points for PCA: {core_pts.shape[0]} / {len(pts)} ({core_pts.shape[0]/len(pts)*100:.1f}%)")
        
        # Weighted Center
        total_weight = core_scores.sum()
        weighted_center = (core_pts * core_scores.unsqueeze(1)).sum(dim=0) / total_weight
        
        print(f"Estimated Car Center: {weighted_center.cpu().numpy()}")
        
        # Weighted Covariance Matrix
        centered_pts = core_pts - weighted_center
        weighted_centered = centered_pts * core_scores.unsqueeze(1)
        cov_matrix = (weighted_centered.T @ centered_pts) / total_weight
        
        # Eigen Decomposition (PCA)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # 축 정의 (eigenvalues는 오름차순: e[0] < e[1] < e[2])
        up_vector = eigenvectors[:, 0]       # Smallest (Height)
        side_vector = eigenvectors[:, 1]     # Middle (Width)
        forward_vector = eigenvectors[:, 2]  # Largest (Length)
        
        # Up vector가 양수 방향을 향하도록 조정 (Y-up 가정)
        if up_vector[1] < 0:
            up_vector *= -1
        
        print(f"Forward Vector: {forward_vector.cpu().numpy()}")
        print(f"Up Vector: {up_vector.cpu().numpy()}")
        
        # 5. 전체 점을 로컬 좌표계로 변환
        rot_matrix = torch.stack([up_vector, side_vector, forward_vector], dim=1)
        xyz_centered = pts - weighted_center
        xyz_aligned = xyz_centered @ rot_matrix
        
        # 6. Bounding Box 계산
        if use_auto_bounds:
            print("Using automatic histogram-based bounds detection...(Without manual sigma threshold)")
            bounds = []
            
            for dim in range(3):
                dim_vals = xyz_aligned[:, dim]

                # if dim == 0: 
                #     # 바닥(Min): 하위 0.5% 지점 (노이즈 한두 개는 무시하고 바퀴 끝점 찾기)
                #     # 천장(Max): 상위 99.5% 지점
                #     bound_min = torch.quantile(dim_vals, 0.005)
                #     bound_max = torch.quantile(dim_vals, 0.995)
                    
                #     # 약간의 여유(Margin)를 줄지 말지 결정 (Tight하게 하려면 Margin 없이)
                #     # 바닥은 여유 없이 딱 맞게, 위쪽만 살짝 여유를 주고 싶다면:
                #     bounds.append((bound_min.item(), bound_max.item()))
                #     continue

                # 히스토그램 생성
                min_v, max_v = dim_vals.min(), dim_vals.max()
                bins = 200
                hist = torch.histc(dim_vals, bins=bins, min=min_v.item(), max=max_v.item())
                
                # Peak density 찾기
                peak_density = hist.max()
                density_thresh = peak_density * 0.05  # Peak의 5%

                # 유효한 bin 찾기
                valid_bins = torch.where(hist > density_thresh)[0]
                
                if len(valid_bins) == 0:
                    # Fallback: 전체 범위 사용
                    bounds.append((min_v.item(), max_v.item()))
                    continue
                
                start_idx = valid_bins[0]
                end_idx = valid_bins[-1]
                
                # Bin 인덱스를 실제 좌표값으로 변환
                step = (max_v - min_v) / bins
                bound_min = min_v + start_idx * step
                bound_max = min_v + (end_idx + 1) * step
                
                # 여유(Padding) 추가 (5%)
                margin = (bound_max - bound_min) * 0.05
                bounds.append((bound_min.item() - margin.item(), bound_max.item() + margin.item()))
            
            # 너무 납작한 차원 방지: 각 dim이 다른 dim의 최소 30% 이상 되도록 보장
            # (주로 높이가 너무 작아지는 것을 방지)
            dim_sizes = [bounds[i][1] - bounds[i][0] for i in range(3)]
            max_size = max(dim_sizes)
            min_ratio = 0.2  # 최소 30%

            print(f"Initial box sizes: Height={dim_sizes[0]:.4f}, Width={dim_sizes[1]:.4f}, Length={dim_sizes[2]:.4f}")
            
            adjusted = False
            for dim in range(3):
                current_size = dim_sizes[dim]
                min_allowed_size = max_size * min_ratio
                
                if current_size < min_allowed_size:
                    # 너무 작은 차원을 확장 (중심 기준으로 양쪽으로)
                    center_val = (bounds[dim][0] + bounds[dim][1]) / 2
                    half_size = min_allowed_size / 2
                    bounds[dim] = (center_val - half_size, center_val + half_size)
                    
                    dim_name = ['Height(Up)', 'Width(Side)', 'Length(Forward)'][dim]
                    print(f"  ⚠ {dim_name} too small ({current_size:.4f} < {min_allowed_size:.4f})")
                    print(f"    → Expanded to {min_allowed_size:.4f} (30% of max dim)")
                    adjusted = True
            
            if not adjusted:
                print(f"  ✓ All dimensions are within reasonable ratios")
            
            print(f"Final Bounds (Local Axis): {bounds}")
        else:
            # Manual sigma-based bounds
            if sigma_thresh is None:
                sigma_thresh = 2.5
            print(f"Using manual sigma threshold: {sigma_thresh}")
            
            high_score_aligned = xyz_aligned[high_score_mask]
            sigmas = torch.std(high_score_aligned, dim=0)
            bounds_vals = sigmas * sigma_thresh
            
            bounds = [
                (-bounds_vals[0].item(), bounds_vals[0].item()),
                (-bounds_vals[1].item(), bounds_vals[1].item()),
                (-bounds_vals[2].item(), bounds_vals[2].item())
            ]
            print(f"Sigma-based Bounds (Local Axis): {bounds}")
        
        # 7. Filtering (Pruning)
        mask = (xyz_aligned[:, 0] >= bounds[0][0]) & (xyz_aligned[:, 0] <= bounds[0][1]) & \
               (xyz_aligned[:, 1] >= bounds[1][0]) & (xyz_aligned[:, 1] <= bounds[1][1]) & \
               (xyz_aligned[:, 2] >= bounds[2][0]) & (xyz_aligned[:, 2] <= bounds[2][1])
        
        num_before = len(pts)
        num_after = mask.sum().item()
        pruning_ratio = (num_before - num_after) / num_before
        
        print(f"Pruning: {num_before} -> {num_after} Gaussians (Pruned: {pruning_ratio*100:.1f}%)")
        
        # 수렴 체크: 이전 iteration과 거의 변화가 없으면 조기 종료
        if prev_count is not None:
            change_ratio = abs(prev_count - num_after) / prev_count
            print(f"Change from prev iteration: {change_ratio*100:.2f}%")
            
            if change_ratio < convergence_threshold:
                print(f"\n{'='*60}")
                print(f"✓ CONVERGED at iteration {iter_num}/{iterations}")
                print(f"  - Change ratio ({change_ratio*100:.2f}%) < threshold ({convergence_threshold*100:.1f}%)")
                print(f"  - Vehicle is well isolated, stopping early")
                print(f"{'='*60}\n")
                # 현재 결과를 최종 결과로 사용
                break
        
        prev_count = num_after
        
        # 다음 iteration을 위해 마스킹된 데이터 저장
        if iter_num < iterations:
            # 다음 iteration을 위해 필터링된 데이터 준비
            current_pts = pts[mask]
            current_opa = raw_opa[mask]
            current_scl = raw_scl[mask]
            
            # raw_vertex도 마스킹 (numpy로 변환 필요)
            mask_np = mask.cpu().numpy()
            new_data = {}
            for prop in raw_vertex.properties:
                new_data[prop.name] = raw_vertex[prop.name][mask_np]
            
            types = [(p.name, raw_vertex[p.name].dtype) for p in raw_vertex.properties]
            new_vertex_data = np.zeros(np.sum(mask_np), dtype=types)
            for name, dtype in types:
                new_vertex_data[name] = new_data[name]
            
            # Subscriptable한 임시 객체 생성 (다음 iteration용)
            class VertexWrapper:
                def __init__(self, properties, data_dict):
                    self.properties = properties
                    self._data = data_dict
                
                def __getitem__(self, key):
                    return self._data[key]
            
            current_vertex = VertexWrapper(raw_vertex.properties, new_data)
    
    # 8. 최종 저장 (마지막 iteration 결과)
    save_ply(raw_vertex, mask, output_path)
    
    # 9. 메타데이터 저장
    if save_meta:
        # TXT 저장
        meta_txt_path = output_path.replace('.ply', '_metadata.txt')
        save_metadata(
            weighted_center.cpu().numpy(),
            eigenvectors.cpu().numpy(),
            eigenvalues.cpu().numpy(),
            meta_txt_path
        )
        
        # JSON 저장 (웹 시각화용)
        meta_json_path = output_path.replace('.ply', '_pose.json')
        pose_info = {
            "center": weighted_center.tolist(),
            "forward": forward_vector.tolist(),
            "up": up_vector.tolist(),
            "side": side_vector.tolist(),
            "bbox_local": {
                "height": [bounds[0][0], bounds[0][1]],
                "width": [bounds[1][0], bounds[1][1]],
                "length": [bounds[2][0], bounds[2][1]]
            },
            "eigenvalues": eigenvalues.tolist(),
            "method": "auto_histogram" if use_auto_bounds else f"sigma_{sigma_thresh}"
        }
        
        # [NEW] Axis alignment verification
        # Apply hemisphere and CLIP-based corrections
        pose_info = align_axes(
            pose_info, 
            colmap_path=colmap_path,
            image_folder=image_folder,
            device=str(device)
        )
        
        with open(meta_json_path, 'w') as f:
            json.dump(pose_info, f, indent=4)
        print(f"Pose info saved to: {meta_json_path}")

# --- 실행 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CUDA-accelerated Mahalanobis distance based point cloud pruning for vehicles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic bounds detection (recommended)
  python mahalanobis_pruning.py --input point_cloud.ply --output cleaned.ply
  
  # Multiple iterations for better results
  python mahalanobis_pruning.py -i point_cloud.ply -o cleaned.ply -n 3
  
  # Manual sigma threshold with iterations
  python mahalanobis_pruning.py -i data.ply -o result.ply --manual -t 2.5 -n 2
  
  # CPU only mode
  python mahalanobis_pruning.py -i input.ply -o output.ply --device cpu
  
  # No metadata files
  python mahalanobis_pruning.py -i input.ply -o output.ply --no-meta

Iterations:
  - More iterations = progressively better pruning results
  - Each iteration refines the bounding box based on previous results
  - Recommended: 2-5 iterations for most cases
  - Diminishing returns after 5+ iterations

Threshold Tips (manual mode only):
  - Higher values (3.0+): Include more background
  - Lower values (1.5-2.0): Tighter fit, may clip vehicle parts
  - Default (2.5): Balanced for most cases
  
Automatic Mode:
  - Uses histogram-based density analysis
  - No manual threshold needed
  - Generally more robust for varying scenes
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input PLY file path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output PLY file path')
    parser.add_argument('-t', '--threshold', type=float, default=2.0,
                        help='Sigma threshold for pruning (only used with --manual, default: 2.0)')
    parser.add_argument('-n', '--iterations', type=int, default=1,
                        help='Number of iterative pruning passes (default: 1, recommended: 2-5)')
    parser.add_argument('--convergence', type=float, default=0.01,
                        help='Early stop if change ratio < this (default: 0.01 = 1%%)')
    parser.add_argument('--min-score', type=float, default=0.7,
                        help='Minimum score percentile for core points (default: 0.7 = top 30%%)')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual sigma-based bounds instead of automatic histogram detection')
    parser.add_argument('--no-meta', action='store_true',
                        help='Do not save metadata files (txt and json)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu). Auto-detect if not specified.')
    parser.add_argument('--colmap-path', type=str, default=None,
                        help='Path to COLMAP sparse folder (e.g., data/project/sparse/0) for axis alignment verification')
    parser.add_argument('--image-folder', type=str, default=None,
                        help='Path to images folder for CLIP-based forward/backward axis verification')
    
    args = parser.parse_args()
    
    auto_prune_vehicle(
        input_path=args.input,
        output_path=args.output,
        sigma_thresh=args.threshold,
        save_meta=not args.no_meta,
        use_auto_bounds=not args.manual,
        device=args.device,
        iterations=args.iterations,
        colmap_path=args.colmap_path,
        image_folder=args.image_folder,
        convergence_threshold=args.convergence,
        min_score_percentile=args.min_score
    )