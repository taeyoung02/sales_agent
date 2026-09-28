"""
CF3 Heatmap Generator
Generates heatmap PLY files based on text queries using CF3 feature vectors
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement
from typing import Optional, Dict, Tuple
from datetime import datetime
import warnings

try:
    from sklearn.cluster import DBSCAN
    from scipy.linalg import eigh
    from sklearn.neighbors import NearestNeighbors
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("Warning: sklearn or scipy not available. Advanced clustering features disabled.")

warnings.filterwarnings("ignore", category=UserWarning)

# Exponential coloring parameter (fixed per spec)
ALPHA = 3.0
DEVICE = os.getenv("DEVICE", "cpu")

# Parts list for generating mean vector
PARTS_LIST = [
    "front bumper", "rear bumper", "car hood", "car door", "car roof", 
    "fender", "wheel", "headlight", "taillight", "windshield", 
    "rear window", "side window", "windshield wiper", "side mirror", 
    "car door handle", "license plate", "emblem", "grille", "exhaust pipe"
]


class Autoencoder(nn.Module):
    """Autoencoder model for CF3 feature extraction"""

    def __init__(self, input_size, latent_size=[128, 64, 32, 16]):
        super().__init__()
        # Create encoder layers: 512D -> 3D
        encoder_layers = []
        input_dim = input_size
        for i, dim in enumerate(latent_size):
            encoder_layers.append(nn.Linear(input_dim, dim, bias=True))
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = dim
        # Latent layer (3 dim)
        encoder_layers.append(nn.Linear(input_dim, 3, bias=True))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers: 3D -> 512D
        decoder_layers = []
        input_dim = 3
        for i, dim in enumerate(reversed(latent_size)):
            decoder_layers.append(nn.Linear(input_dim, dim, bias=True))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = dim
        # Final layer
        decoder_layers.append(nn.Linear(input_dim, input_size, bias=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x: 512D
        # return (decoded 512D, encoded 3D)
        return self.decoder(self.encoder(x)), self.encoder(x)


class OpenCLIPTextEncoder:
    """CLIP text encoder wrapper - matches cf3_demo.py implementation"""

    def __init__(self, device="cpu"):
        try:
            import open_clip

            # Use same model as cf3_demo.py: ViT-B-32 with openai
            self.clip_model_type = "ViT-B-32"
            self.clip_model_pretrained = "openai"

            model, _, _ = open_clip.create_model_and_transforms(
                self.clip_model_type,
                pretrained=self.clip_model_pretrained,
                precision="fp16" if device != "cpu" else None,  # fp16 only for GPU
            )
            model.eval()
            self.model = model.to(device)
            self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
            self.device = device
        except ImportError:
            raise ImportError(
                "open-clip-torch is required. Install with: pip install open-clip-torch"
            )

    @torch.no_grad()
    def encode(self, texts):
        """Encode text to feature vector - matches cf3_demo.py signature"""
        # Handle both string and list inputs
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(texts).to(self.device)
        embeds = self.model.encode_text(tokens)
        return embeds / embeds.norm(dim=-1, keepdim=True)  # L2 정규화


# Singleton instances for caching
_autoencoder_cache: Optional[Autoencoder] = None
_clip_encoder_cache: Optional[OpenCLIPTextEncoder] = None
_cached_ae_path: Optional[str] = None


def load_autoencoder(ae_path: str, feature_size: int = 512) -> Autoencoder:
    """Load autoencoder model with caching - matches cf3_demo.py"""
    global _autoencoder_cache, _cached_ae_path

    if _autoencoder_cache is not None and _cached_ae_path == ae_path:
        return _autoencoder_cache

    print(f"Loading autoencoder from {ae_path}...")
    ae = Autoencoder(feature_size).to("cpu")
    state = torch.load(ae_path, map_location="cpu")
    ae.load_state_dict(state)
    ae.eval()

    _autoencoder_cache = ae
    _cached_ae_path = ae_path
    print("Autoencoder loaded successfully")
    return ae


def load_clip_encoder(device: str = "cpu") -> OpenCLIPTextEncoder:
    """Load CLIP encoder with caching - matches cf3_demo.py"""
    global _clip_encoder_cache

    if _clip_encoder_cache is not None:
        return _clip_encoder_cache

    print(f"Loading CLIP encoder (device: {device})...")
    clip = OpenCLIPTextEncoder(device)
    _clip_encoder_cache = clip
    print("CLIP encoder loaded successfully")
    return clip


def generate_heatmap(
    cf3_path: str,
    reference_path: str,
    query: str,
    ae_model_path: Optional[str],
    output_dir: str,
    feature_size: int = 512,
    use_autoencoder: bool = True,
    vehicle_center: Optional[Tuple[float, float, float]] = None,
    mean_vector_path: Optional[str] = None,
    skip_camera_calculation: bool = False,
) -> tuple[str, Optional[tuple[float, float, float]], Optional[tuple[float, float, float]], Optional[tuple[float, float, float]], Optional[tuple[float, float, float]]]:
    """
    Generate heatmap PLY file for a single query

    Args:
        cf3_path: Path to CF3 PLY file (with feature vectors)
        reference_path: Path to reference PLY file (for geometry)
        query: Text query for heatmap generation
        ae_model_path: Path to autoencoder model file (required if use_autoencoder=True)
        output_dir: Directory to save output PLY file
        feature_size: Feature vector size (default: 512)
        use_autoencoder: Whether to use autoencoder to decode 3D features to 512D (default: True)
        vehicle_center: Vehicle center position for camera orientation
        mean_vector_path: Path to mean vector file for semantic centering (optional, auto-enables semantic centering if provided)
        skip_camera_calculation: If True, skip surface normal and camera calculation

    Returns:
        Tuple of (output_path, target_position, surface_normal, camera_position, surface_center)
        output_path: Path to generated PLY file
        target_position: (x, y, z) coordinates of highest similarity point, or None if not found
        surface_normal: Surface normal vector, or None if skip_camera_calculation=True
        camera_position: Camera position, or None if skip_camera_calculation=True
        surface_center: Surface center position, or None if skip_camera_calculation=True
    """
    # Load CLIP encoder
    clip = load_clip_encoder("cpu")

    # Load CF3 PLY file
    ply = PlyData.read(cf3_path)["vertex"].data

    if use_autoencoder:
        # Autoencoder 사용
        # 3D features (f_dc_0, f_dc_1, f_dc_2) -> decode to 512D
        if ae_model_path is None:
            raise ValueError("ae_model_path is required when use_autoencoder=True")

        ae = load_autoencoder(ae_model_path, feature_size)

        # vstack: vertical stack (열 방향으로 데이터 저장)
        # T: 전치 행렬 (열 방향으로 데이터 저장)
        # arr: 2D array
        arr = np.vstack(
            [ply[p] for p in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]]
        ).T
        cf3 = torch.from_numpy(arr).float().to(DEVICE)  # 텐서 변환
        print(f"CF3 data shape: {cf3.shape}")

        # Extract features using autoencoder decoder
        with torch.no_grad():
            cf3_feat = ae.decoder(cf3[:, 3:])
        print(f"CF3 features shape (decoded): {cf3_feat.shape}")

        # Store original cf3 for position extraction
        cf3_coords = cf3[:, :3]
    else:
        # Autoencoder 미사용
        # 512D features 직접 읽기
        print("Autoencoder 미사용")

        # Standard PLY columns to exclude (geometry, transform, and 3D features)
        standard_cols = {
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "opacity",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",  # 3D features (used with autoencoder)
            "f_rest_0",  # Spherical harmonics rest terms (not semantic features)
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "contribution",  # Additional metadata
            "feature_var",  # Additional metadata
        }

        # Add all f_rest_* columns to standard (they're spherical harmonics, not semantic features)
        for i in range(100):  # Cover f_rest_0 to f_rest_99
            standard_cols.add(f"f_rest_{i}")

        # Get all available column names
        all_cols = list(ply.dtype.names)
        print(f"All PLY columns ({len(all_cols)}): {all_cols[:10]}...")

        # Find feature columns by excluding standard columns
        feature_cols = [col for col in all_cols if col not in standard_cols]

        # Try to find columns that match common patterns
        # Pattern 1: semantic_0, semantic_1, ..., semantic_N
        semantic_pattern_cols = sorted(
            [col for col in feature_cols if col.startswith("semantic_")],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1,
        )

        # Pattern 2: feat_0, feat_1, ..., feat_N
        feat_pattern_cols = sorted(
            [col for col in feature_cols if col.startswith("feat_")],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1,
        )

        # Pattern 3: f_0, f_1, ..., f_N (generic feature naming)
        f_pattern_cols = sorted(
            [
                col
                for col in feature_cols
                if col.startswith("f_") and col not in ["f_dc_0", "f_dc_1", "f_dc_2"]
            ],
            key=lambda x: (
                int(x.split("_")[1])
                if len(x.split("_")) > 1 and x.split("_")[1].isdigit()
                else -1
            ),
        )

        # Choose the best matching pattern
        if len(semantic_pattern_cols) >= feature_size:
            available_cols = semantic_pattern_cols[:feature_size]
            print(
                f"Using semantic_* pattern: found {len(semantic_pattern_cols)} columns, using first {feature_size}"
            )
        elif len(feat_pattern_cols) >= feature_size:
            available_cols = feat_pattern_cols[:feature_size]
            print(
                f"Using feat_* pattern: found {len(feat_pattern_cols)} columns, using first {feature_size}"
            )
        elif len(f_pattern_cols) >= feature_size:
            available_cols = f_pattern_cols[:feature_size]
            print(
                f"Using f_* pattern: found {len(f_pattern_cols)} columns, using first {feature_size}"
            )
        elif len(feature_cols) >= feature_size:
            # Use first N feature columns if no clear pattern
            available_cols = feature_cols[:feature_size]
            print(
                f"Using first {feature_size} non-standard columns from {len(feature_cols)} available"
            )
        else:
            # List available columns for debugging
            raise ValueError(
                f"Could not find {feature_size}D feature columns in PLY file.\n"
                f"Found {len(feature_cols)} non-standard columns.\n"
                f"Available non-standard columns: {feature_cols[:30]}...\n"
                f"Please ensure the PLY file contains {feature_size} feature columns."
            )

        if len(available_cols) != feature_size:
            print(
                f"Warning: Using {len(available_cols)} feature columns, expected {feature_size}"
            )

        # Read coordinates and features
        coord_arr = np.vstack([ply[p] for p in ["x", "y", "z"]]).T
        feat_arr = np.vstack([ply[p] for p in available_cols]).T

        cf3_coords = torch.from_numpy(coord_arr).float().to("cpu")
        cf3_feat = torch.from_numpy(feat_arr).float().to("cpu")
        print(f"CF3 coordinates shape: {cf3_coords.shape}")
        print(f"CF3 features shape (direct): {cf3_feat.shape}")

    # Load or generate mean vector for semantic centering (auto-enabled if mean_vector_path provided)
    mean_vector = None
    if mean_vector_path:
        if Path(mean_vector_path).exists():
            # Load existing mean vector
            try:
                mean_vector = torch.load(mean_vector_path, map_location="cpu")
                print(f"✅ Loaded mean vector from: {mean_vector_path}")
                print(f"   Mean vector shape: {mean_vector.shape}")
                print(f"✅ Semantic centering ENABLED (mean_vector_path provided)")
            except Exception as e:
                print(f"⚠️  Could not load mean vector: {e}")
                mean_vector = None
        else:
            # Auto-generate mean vector if it doesn't exist
            print(f"⚠️  Mean vector not found at {mean_vector_path}, auto-generating...")
            try:
                # Determine feature field path from cf3_path
                feature_field_path = cf3_path  # Use CF3 as feature field
                
                mean_vector = generate_mean_vector(
                    feature_field_path=feature_field_path,
                    ae_model_path=ae_model_path,
                    output_path=mean_vector_path,
                    parts_list=PARTS_LIST,
                    feature_size=feature_size
                )
                print(f"✅ Auto-generated and saved mean vector to: {mean_vector_path}")
                print(f"✅ Semantic centering ENABLED (mean vector auto-generated)")
            except Exception as e:
                print(f"⚠️  Failed to auto-generate mean vector: {e}")
                mean_vector = None
    else:
        print("ℹ️  Semantic centering DISABLED (no mean_vector_path provided)")

    # Apply semantic centering to features (only if mean_vector is available)
    if mean_vector is not None:
        print("🔧 Applying semantic centering to features...")
        cf3_feat = apply_semantic_centering(cf3_feat, mean_vector)

    # Encode query text (no preprocessing - raw query to match feature field)
    print(f"Encoding query: '{query}'")
    txt_feat = clip.encode(query).squeeze().to(torch.float32)
    print(f"Text feature shape: {txt_feat.shape}")

    # Ensure txt_feat is 1D (remove batch dimension if present)
    if txt_feat.dim() > 1:
        txt_feat = txt_feat.squeeze(0)

    # Apply semantic centering to query if mean vector is available
    if mean_vector is not None:
        print("🔧 Applying semantic centering to query...")
        txt_feat = apply_semantic_centering(txt_feat.unsqueeze(0), mean_vector).squeeze(0)

    # Calculate similarity (cosine similarity)
    if mean_vector is not None:
        # Features are already normalized by apply_semantic_centering
        sim_map = torch.einsum("nc,c->n", cf3_feat, txt_feat)
    else:
        sim_map = torch.einsum(
            "nc,c->n", F.normalize(cf3_feat, dim=1), F.normalize(txt_feat, dim=0)
        )
    print(
        f"Similarity map shape: {sim_map.shape}, min: {sim_map.min():.4f}, max: {sim_map.max():.4f}"
    )

    # Find point with highest similarity for camera positioning
    target_position: Optional[tuple[float, float, float]] = None
    try:
        max_sim_idx = torch.argmax(sim_map).item()
        print(f"[max 유사도 인덱스] {max_sim_idx}")
        if 0 <= max_sim_idx < len(cf3_coords):
            # Extract 3D coordinates
            x = float(cf3_coords[max_sim_idx, 0].item())
            y = float(cf3_coords[max_sim_idx, 1].item())
            z = float(cf3_coords[max_sim_idx, 2].item())
            target_position = (x, y, z)
            print(
                f"Target position (highest similarity at index {max_sim_idx}): ({x:.3f}, {y:.3f}, {z:.3f})"
            )
    except Exception as e:
        print(f"Warning: Could not extract target position: {e}")
        target_position = None

    # Normalize similarity scores to [0, 1] - exactly match cf3_demo.py
    sims = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-9)
    sims_np = sims.cpu().numpy()
    print(f"Normalized similarity range: [{sims_np.min():.4f}, {sims_np.max():.4f}]")

    # Apply exponential mapping for better visualization - exactly match cf3_demo.py
    alpha = ALPHA
    denom = np.exp(alpha) - 1.0
    if denom <= 0:
        denom = 1e-9
    sims_mapped = np.expm1(alpha * sims_np) / denom
    print(
        f"Mapped similarity range: [{sims_mapped.min():.4f}, {sims_mapped.max():.4f}]"
    )

    # Map to colors using jet colormap - exactly match cf3_demo.py
    rgba = cm.get_cmap("jet")(sims_mapped)
    print(f"RGBA shape: {rgba.shape}")

    # Load reference PLY - use cf3_path (feature field) as base to match spark behavior
    print(f"Loading reference PLY from: {cf3_path}")
    ref_data = PlyData.read(cf3_path)["vertex"].data
    colored = ref_data.copy()

    # Apply colors to f_dc channels (spherical harmonics DC term) - exactly match cf3_demo.py
    for idx in range(3):
        colored[f"f_dc_{idx}"] = (rgba[:, idx] * 2.0) - 1.0
    print("Colors applied to f_dc channels")

    # Create output PLY file
    vert = PlyElement.describe(colored, name="vertex")
    ply_out = PlyData([vert], text=False)

    # Generate output filename
    safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_filename = f"{safe_query}_heatmap.ply"
    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write PLY file
    ply_out.write(str(output_path))

    # Calculate surface normal and camera position if not skipped
    surface_normal = None
    camera_position = None
    surface_center = None
    
    if not skip_camera_calculation and CLUSTERING_AVAILABLE:
        print("\n=== Calculating surface normal and camera position ===")
        coords_np = cf3_coords.cpu().numpy()
        surface_normal, camera_position, surface_center = calculate_surface_normal_and_camera(
            coords=coords_np,
            similarities=sims_np,
            ply_data=ref_data,
            threshold_percentile=95.0,
            vehicle_center=vehicle_center,
            min_camera_distance=1.0,
            max_camera_distance=3.0
        )

    return str(output_path), target_position, surface_normal, camera_position, surface_center


def generate_heatmap_target_position_only(
    cf3_path: str,
    query: str,
    ae_model_path: Optional[str],
    feature_size: int = 512,
    use_autoencoder: bool = True,
) -> Optional[tuple[float, float, float]]:
    """
    Generate only target_position for camera positioning (production mode)
    Does not generate PLY file - only calculates the position of highest similarity

    Args:
        cf3_path: Path to CF3 PLY file (with feature vectors)
        query: Text query for heatmap generation
        ae_model_path: Path to autoencoder model file (required if use_autoencoder=True)
        feature_size: Feature vector size (default: 512)
        use_autoencoder: Whether to use autoencoder to decode 3D features to 512D (default: True)

    Returns:
        target_position: (x, y, z) coordinates of highest similarity point, or None if not found
    """
    # Load CLIP encoder
    clip = load_clip_encoder("cpu")

    # Load CF3 PLY file
    ply = PlyData.read(cf3_path)["vertex"].data

    if use_autoencoder:
        # Autoencoder 사용
        if ae_model_path is None:
            raise ValueError("ae_model_path is required when use_autoencoder=True")

        ae = load_autoencoder(ae_model_path, feature_size)

        arr = np.vstack(
            [ply[p] for p in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]]
        ).T
        cf3 = torch.from_numpy(arr).float().to(DEVICE)

        # Extract features using autoencoder decoder
        with torch.no_grad():
            cf3_feat = ae.decoder(cf3[:, 3:])

        # Store original cf3 for position extraction
        cf3_coords = cf3[:, :3]
    else:
        # Autoencoder 미사용 - 512D features 직접 읽기
        standard_cols = {
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "opacity",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "contribution",
            "feature_var",
        }

        for i in range(100):
            standard_cols.add(f"f_rest_{i}")

        all_cols = list(ply.dtype.names)
        feature_cols = [col for col in all_cols if col not in standard_cols]

        semantic_pattern_cols = sorted(
            [col for col in feature_cols if col.startswith("semantic_")],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1,
        )
        feat_pattern_cols = sorted(
            [col for col in feature_cols if col.startswith("feat_")],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else -1,
        )
        f_pattern_cols = sorted(
            [
                col
                for col in feature_cols
                if col.startswith("f_") and col not in ["f_dc_0", "f_dc_1", "f_dc_2"]
            ],
            key=lambda x: (
                int(x.split("_")[1])
                if len(x.split("_")) > 1 and x.split("_")[1].isdigit()
                else -1
            ),
        )

        if len(semantic_pattern_cols) >= feature_size:
            available_cols = semantic_pattern_cols[:feature_size]
        elif len(feat_pattern_cols) >= feature_size:
            available_cols = feat_pattern_cols[:feature_size]
        elif len(f_pattern_cols) >= feature_size:
            available_cols = f_pattern_cols[:feature_size]
        elif len(feature_cols) >= feature_size:
            available_cols = feature_cols[:feature_size]
        else:
            raise ValueError(
                f"Could not find {feature_size}D feature columns in PLY file."
            )

        coord_arr = np.vstack([ply[p] for p in ["x", "y", "z"]]).T
        feat_arr = np.vstack([ply[p] for p in available_cols]).T

        cf3_coords = torch.from_numpy(coord_arr).float().to("cpu")
        cf3_feat = torch.from_numpy(feat_arr).float().to("cpu")

    # Encode query text (no preprocessing - raw query to match feature field)
    txt_feat = clip.encode(query).squeeze().to(torch.float32)
    if txt_feat.dim() > 1:
        txt_feat = txt_feat.squeeze(0)

    # Calculate similarity (cosine similarity)
    sim_map = torch.einsum(
        "nc,c->n", F.normalize(cf3_feat, dim=1), F.normalize(txt_feat, dim=0)
    )

    # Find point with highest similarity for camera positioning
    target_position: Optional[tuple[float, float, float]] = None
    try:
        max_sim_idx = torch.argmax(sim_map).item()
        if 0 <= max_sim_idx < len(cf3_coords):
            x = float(cf3_coords[max_sim_idx, 0].item())
            y = float(cf3_coords[max_sim_idx, 1].item())
            z = float(cf3_coords[max_sim_idx, 2].item())
            target_position = (x, y, z)
            print(
                f"Target position (highest similarity at index {max_sim_idx}): ({x:.3f}, {y:.3f}, {z:.3f})"
            )
    except Exception as e:
        print(f"Warning: Could not extract target position: {e}")
        target_position = None

    return target_position


def generate_mean_vector(
    feature_field_path: str,
    ae_model_path: Optional[str],
    output_path: str,
    parts_list: Optional[list] = None,
    feature_size: int = 512
) -> torch.Tensor:
    """
    Generate mean vector from parts list for semantic centering
    
    Args:
        feature_field_path: Path to feature field PLY file
        ae_model_path: Path to autoencoder model (optional, not used for mean vector generation)
        output_path: Path to save mean vector .pt file
        parts_list: List of part names to generate mean vector
        feature_size: Feature vector dimension
    
    Returns:
        Mean vector tensor [D]
    """
    if parts_list is None:
        parts_list = PARTS_LIST
    
    try:
        print(f"Generating mean vector from {len(parts_list)} parts...")
        
        # Load CLIP encoder only (don't need autoencoder for mean vector)
        clip = load_clip_encoder("cpu")
        
        # Encode all parts (no preprocessing - raw part names to match feature field)
        part_features = []
        for part in parts_list:
            txt_feat = clip.encode(part).squeeze().to(torch.float32)
            if txt_feat.dim() > 1:
                txt_feat = txt_feat.squeeze(0)
            part_features.append(txt_feat)
        
        # Stack and compute mean
        part_features_tensor = torch.stack(part_features)
        mean_vector = part_features_tensor.mean(dim=0)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        torch.save(mean_vector, output_path)
        print(f"Mean vector saved to: {output_path}")
        print(f"Mean vector shape: {mean_vector.shape}")
        
        return mean_vector
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate mean vector: {e}")


def apply_semantic_centering(features: torch.Tensor, mean_vector: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply semantic centering to features for better discrimination
    V_new = Normalize(V_old - Mean)
    
    Args:
        features: [N, D] feature tensor
        mean_vector: [D] mean vector (optional)
    
    Returns:
        Centered and normalized features [N, D]
    """
    if mean_vector is None:
        return F.normalize(features, dim=-1)
    
    centered = features - mean_vector
    normalized = F.normalize(centered, dim=-1)
    return normalized


def calculate_surface_normal_and_camera(
    coords: np.ndarray,
    similarities: np.ndarray,
    ply_data,
    threshold_percentile: float = 95.0,
    vehicle_center: Optional[Tuple[float, float, float]] = None,
    min_camera_distance: float = 1.0,
    max_camera_distance: float = 3.0,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """
    Calculate surface normal and camera position from high-similarity points
    
    Args:
        coords: Nx3 array of point coordinates
        similarities: N-length array of similarity scores [0, 1]
        ply_data: PLY vertex data containing opacity and scale
        threshold_percentile: Percentile threshold for selecting high-similarity points
        vehicle_center: Vehicle center position (x, y, z)
        min_camera_distance: Minimum distance from surface (for small features)
        max_camera_distance: Maximum distance from surface (for large features)
    
    Returns:
        Tuple of (surface_normal, camera_position, surface_center)
    """
    if not CLUSTERING_AVAILABLE:
        print("Warning: Clustering not available. Skipping surface normal calculation.")
        return None, None, None
    
    try:
        threshold = np.percentile(similarities, threshold_percentile)
        high_sim_mask = similarities >= threshold
        high_sim_indices = np.where(high_sim_mask)[0]
        
        print(f"\n=== Surface Normal Calculation ===")
        print(f"Threshold (top {100-threshold_percentile}%): {threshold:.4f}")
        print(f"High similarity points: {len(high_sim_indices)}/{len(similarities)}")
        
        if len(high_sim_indices) < 10:
            print("Warning: Too few high-similarity points for reliable normal calculation")
            return None, None, None
        
        high_sim_coords = coords[high_sim_indices]
        high_sim_scores = similarities[high_sim_indices]
        
        # DBSCAN clustering
        # 동적 eps 설정: 데이터 스케일에 따라 조정
        if len(high_sim_coords) > 100:
            # [변경] pdist 대신 KNN 사용: 각 점의 "가장 가까운 이웃까지의 거리"를 구함
            neighbors = NearestNeighbors(n_neighbors=2).fit(high_sim_coords)
            distances, _ = neighbors.kneighbors(high_sim_coords)
            
            # 각 점의 가장 가까운 이웃 거리(distances[:, 1])의 평균 근처를 eps로 잡음
            knn_dist = distances[:, 1]
            # 하위 10%가 아니라 median이나 평균을 기준으로 잡되, 약간 여유를 둠
            eps_dynamic = np.percentile(knn_dist, 70) * 1.5 
            
            # [중요] 최소값 제한(0.1)을 데이터셋 스케일에 맞게 낮춤 (0.02 등으로)
            # 데이터 좌표 범위를 확인해보세요 (예: -1 ~ 1 사이라면 0.1은 엄청 큼)
            eps = max(0.02, min(0.3, eps_dynamic)) 
            
            print(f"  Dynamic DBSCAN eps (KNN based): {eps:.4f}")
            
            # min_samples도 점 개수에 비례하게 조정
            min_pts = max(5, int(len(high_sim_coords) * 0.005)) # 전체 점의 0.5%
            clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(high_sim_coords)
            labels = clustering.labels_
            
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 0:
                best_cluster_label = None
                best_cluster_score = -1
                
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_score_sum = high_sim_scores[cluster_mask].sum()
                    cluster_size = cluster_mask.sum()
                    
                    print(f"  Cluster {label}: {cluster_size} points, score sum: {cluster_score_sum:.3f}")
                    
                    if cluster_score_sum > best_cluster_score:
                        best_cluster_score = cluster_score_sum
                        best_cluster_label = label
                
                cluster_mask = labels == best_cluster_label
                high_sim_coords = high_sim_coords[cluster_mask]
                high_sim_scores = high_sim_scores[cluster_mask]
                high_sim_indices = high_sim_indices[cluster_mask]
                
                noise_count = (labels == -1).sum()
                print(f"Clustering: {len(unique_labels)} clusters found ({noise_count} noise points)")
                print(f"Best cluster (label {best_cluster_label}): {len(high_sim_coords)} points, total score: {best_cluster_score:.3f}")
        
        # Extract opacity and scale for weighting
        try:
            opacity = ply_data['opacity'][high_sim_indices]
            opacity = np.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0)
            opacity = np.clip(opacity, 0.0, 1.0)
            
            if 'scale_0' in ply_data.dtype.names:
                scale = np.stack([
                    ply_data['scale_0'][high_sim_indices],
                    ply_data['scale_1'][high_sim_indices],
                    ply_data['scale_2'][high_sim_indices]
                ], axis=1)
                scale_magnitude = np.linalg.norm(scale, axis=1)
                scale_magnitude = np.nan_to_num(scale_magnitude, nan=1.0, posinf=10.0, neginf=0.1)
                scale_magnitude = np.clip(scale_magnitude, 0.01, 100.0)
            else:
                scale_magnitude = np.ones(len(high_sim_indices))

            combined_scores = high_sim_scores * opacity
            combined_scores = np.nan_to_num(combined_scores, nan=0.0, posinf=1.0, neginf=0.0)
            combined_scores = np.maximum(combined_scores, 1e-6)
            
            print(f"Score range - opacity: [{opacity.min():.3f}, {opacity.max():.3f}], "
                  f"scale: [{scale_magnitude.min():.3f}, {scale_magnitude.max():.3f}], "
                  f"combined: [{combined_scores.min():.3e}, {combined_scores.max():.3e}]")
            
            median_scale = np.median(scale_magnitude)
            scale_normalized = np.clip((median_scale - 5.0) / 10.0, 0.0, 1.0)
            camera_distance = min_camera_distance + (max_camera_distance - min_camera_distance) * scale_normalized
            
            print(f"Camera distance - median scale: {median_scale:.3f}")
            print(f"Camera distance - normalized: {scale_normalized:.3f}, final distance: {camera_distance:.3f}m")
            
        except Exception as e:
            print(f"Warning: Could not extract opacity/scale, using similarity only: {e}")
            combined_scores = high_sim_scores
            combined_scores = np.maximum(combined_scores, 1e-6)
            camera_distance = max_camera_distance
        
        # Calculate weighted center
        weighted_coords = high_sim_coords.T * combined_scores
        weights_sum = combined_scores.sum()
        
        if weights_sum < 1e-9:
            print("Warning: Total weight is too small")
            return None, None, None
            
        mean = weighted_coords.sum(axis=1) / weights_sum
        
        if not np.all(np.isfinite(mean)):
            print("Warning: Mean contains inf/NaN")
            return None, None, None
        
        centered_coords = high_sim_coords.T - mean[:, np.newaxis]
        
        sqrt_weights = np.sqrt(combined_scores)
        sqrt_weights = np.nan_to_num(sqrt_weights, nan=0.0, posinf=1.0, neginf=0.0)
        weighted_centered = centered_coords * sqrt_weights
        
        # Compute covariance matrix
        cov_matrix = (weighted_centered @ weighted_centered.T) / len(high_sim_indices)
        
        if not np.all(np.isfinite(cov_matrix)):
            print("Warning: Covariance matrix contains inf/NaN")
            return None, None, None
        
        print(f"Covariance matrix shape: {cov_matrix.shape}")
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov_matrix, check_finite=True)
        
        print(f"Eigenvalues: {eigenvalues}")
        
        # PCA Normal (가장 작은 고유값의 벡터 - 가장 납작한 방향)
        pca_normal = eigenvectors[:, 0]
        
        # Radial Normal (차량 중심 -> 부품 중심 벡터)
        if vehicle_center is not None:
            radial_normal = mean - np.array(vehicle_center)
            radial_normal_norm = np.linalg.norm(radial_normal)
            if radial_normal_norm > 1e-6:
                radial_normal /= radial_normal_norm
            else:
                radial_normal = pca_normal
        else:
            radial_normal = pca_normal
        
        # Use PCA normal as final normal
        final_normal = pca_normal
        
        # 1. 방향 통일 (Radial과 같은 반구 방향으로 정렬)
        if np.dot(final_normal, radial_normal) < 0:
            final_normal = -final_normal
        
        # 2. 법선이 바깥쪽(Outward)을 향하도록 강제
        if vehicle_center is not None:
            # 표면에서 차량 중심으로 향하는 벡터 (안쪽 벡터)
            inward_vector = np.array(vehicle_center) - mean
            
            # 법선이 안쪽을 향하고 있다면 (내적 > 0), 바깥쪽으로 뒤집는다
            if np.dot(final_normal, inward_vector) > 0:
                final_normal = -final_normal
                print(f"  [Direction Fix] Normal flipped to point outward")
        
        surface_normal = tuple(final_normal.astype(float))
        camera_position = tuple((mean + final_normal * camera_distance).astype(float))
        surface_center = tuple(mean.astype(float))
        
        print(f"Surface normal: {surface_normal}")
        print(f"Surface center: {surface_center}")
        print(f"Camera position: {camera_position}")
        print(f"Camera distance: {camera_distance:.3f}m")
        
        return surface_normal, camera_position, surface_center
        
    except Exception as e:
        print(f"Error calculating surface normal: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_query_preset(
    dataset_path: str,
    query: str
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Load query preset from point_cloud_pruned_pose.json
    
    Args:
        dataset_path: Path to dataset directory
        query: Text query to look up
    
    Returns:
        Tuple of (surface_normal, camera_position, surface_center) if found, else None
    """
    pose_file = Path(dataset_path) / "point_cloud_pruned_pose.json"
    
    if not pose_file.exists():
        print(f"[Preset] Pose file not found: {pose_file}")
        return None
    
    try:
        with open(pose_file, 'r') as f:
            data = json.load(f)
        
        query_presets = data.get("query_presets", {})
        
        if query in query_presets:
            preset = query_presets[query]
            surface_normal = tuple(preset["surface_normal"])
            camera_position = tuple(preset["camera_position"])
            surface_center = tuple(preset["surface_center"])
            
            timestamp = preset.get("timestamp", "unknown")
            print(f"[Preset] Found preset for '{query}' (saved: {timestamp})")
            print(f"[Preset]   Surface normal: {surface_normal}")
            print(f"[Preset]   Camera position: {camera_position}")
            print(f"[Preset]   Surface center: {surface_center}")
            
            return (surface_normal, camera_position, surface_center)
        else:
            print(f"[Preset] No preset found for '{query}'")
            return None
            
    except Exception as e:
        print(f"[Preset] Error loading preset: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_query_preset(
    dataset_path: str,
    query: str,
    surface_normal: Tuple[float, float, float],
    camera_position: Tuple[float, float, float],
    surface_center: Tuple[float, float, float]
) -> bool:
    """
    Save query preset to point_cloud_pruned_pose.json
    
    Args:
        dataset_path: Path to dataset directory
        query: Text query as key
        surface_normal: Surface normal vector
        camera_position: Camera position
        surface_center: Surface center position
    
    Returns:
        True if saved successfully, False otherwise
    """
    pose_file = Path(dataset_path) / "point_cloud_pruned_pose.json"
    
    try:
        if pose_file.exists():
            with open(pose_file, 'r') as f:
                data = json.load(f)
        else:
            print(f"[Preset] Warning: Pose file not found, creating new structure")
            data = {}
        
        if "query_presets" not in data:
            data["query_presets"] = {}
        
        data["query_presets"][query] = {
            "surface_normal": list(surface_normal),
            "camera_position": list(camera_position),
            "surface_center": list(surface_center),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(pose_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"[Preset] Saved preset for '{query}' to {pose_file}")
        return True
        
    except Exception as e:
        print(f"[Preset] Error saving preset: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_initial_camera_position(
    dataset_path: str,
    camera_position: Tuple[float, float, float]
) -> bool:
    """
    Save initial camera position to point_cloud_pruned_pose.json
    
    Args:
        dataset_path: Path to dataset directory
        camera_position: Initial camera position (x, y, z)
    
    Returns:
        True if saved successfully, False otherwise
    """
    pose_file = Path(dataset_path) / "point_cloud_pruned_pose.json"
    
    try:
        if pose_file.exists():
            with open(pose_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        data["initial_camera_position"] = list(camera_position)
        
        with open(pose_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"[Initial Camera] Saved initial camera position to {pose_file}")
        return True
        
    except Exception as e:
        print(f"[Initial Camera] Error saving initial camera position: {e}")
        import traceback
        traceback.print_exc()
        return False
