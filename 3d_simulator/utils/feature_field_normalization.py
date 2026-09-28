"""
Feature Field Normalization for CF3 3D Point Clouds

이 스크립트는 CF3 Autoencoder 기반 Feature Field의 정규화를 수행합니다.
'object', 'car' 같은 dominant한 공통 신호를 제거하여
'right front door', 'wheel' 같은 세부 부품 쿼리의 정확도를 향상시킵니다.

중요: CF3는 Autoencoder로 압축된 3차원 특징(f_0, f_1, f_2)을 사용합니다.
이 스크립트는:
1. PLY에서 압축된 3D features를 읽고
2. Autoencoder Decoder로 고차원 CLIP space로 복원 (e.g., 512-dim)
3. 고차원 공간에서 Mean Subtraction 수행
4. 정규화된 고차원 features를 Autoencoder Encoder로 다시 압축
5. 압축된 정규화 features를 PLY에 저장

전략: Feature Space Whitening (PCA-based Mean Subtraction)
- 디코딩된 Feature Vector에서 전체 평균 벡터(Global Mean)를 빼서
  공통적으로 깔려있던 "Car/Object" 성분을 제거합니다.

사용법:
    python feature_field_normalization.py \
        -i source/toycar/point_cloud.ply \
        -o source/toycar/point_cloud_normalized.ply \
        -a source/toycar/autoencoder.pth \
        --method mean_subtraction
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    CF3 Autoencoder for feature compression/decompression
    Matches the actual CF3 architecture from compact_feature_field.py
    
    Based on error analysis, the actual architecture is:
    Encoder: 512 -> 128 -> 64 -> 32 -> ... -> latent_dim (with more layers)
    Decoder: latent_dim -> 16 -> 32 -> 64 -> ... -> 512 (with more layers)
    """
    def __init__(self, input_dim=512, latent_dim=3):
        super().__init__()
        # Encoder: progressive dimensionality reduction
        # 512 -> 128 -> 64 -> 32 -> 16 -> latent_dim
        # Note: bias=False to match CF3 training configuration
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),  # encoder.0
            nn.ReLU(),                              # encoder.1
            nn.Linear(128, 64, bias=False),         # encoder.2
            nn.ReLU(),                              # encoder.3
            nn.Linear(64, 32, bias=False),          # encoder.4
            nn.ReLU(),                              # encoder.5
            nn.Linear(32, 16, bias=False),          # encoder.6
            nn.ReLU(),                              # encoder.7
            nn.Linear(16, latent_dim, bias=False)   # encoder.8
        )
        
        # Decoder: progressive dimensionality expansion
        # latent_dim -> 16 -> 32 -> 64 -> 128 -> 512
        # Note: bias=False to match CF3 training configuration
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16, bias=False),  # decoder.0
            nn.ReLU(),                              # decoder.1
            nn.Linear(16, 32, bias=False),          # decoder.2
            nn.ReLU(),                              # decoder.3
            nn.Linear(32, 64, bias=False),          # decoder.4
            nn.ReLU(),                              # decoder.5
            nn.Linear(64, 128, bias=False),         # decoder.6
            nn.ReLU(),                              # decoder.7
            nn.Linear(128, input_dim, bias=False)   # decoder.8
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def normalize_ply_features(
    input_path: str,
    output_path: str,
    autoencoder_path: str,
    feature_prefix: str = "f_",
    method: str = "mean_subtraction",
    device: str = "cuda"
) -> None:
    """
    CF3 Autoencoder 기반 PLY Feature Normalization
    
    Args:
        input_path: 원본 PLY 파일 경로 (압축된 f_0, f_1, f_2 포함)
        output_path: 저장할 PLY 파일 경로
        autoencoder_path: CF3 Autoencoder 모델 파일 (.pth)
        feature_prefix: Feature 속성 이름의 접두사 (기본: 'f_')
        method: 'mean_subtraction' (평균 제거) 또는 'whitening' (평균+분산 정규화)
        device: 'cuda' 또는 'cpu'
    
    Process:
        1. Load compressed features (f_0, f_1, f_2) from PLY
        2. Decode to high-dimensional CLIP space (512-dim) using autoencoder
        3. Normalize in high-dim space (mean subtraction)
        4. Re-encode normalized features back to 3D
        5. Save to output PLY
    """
    
    print(f"=" * 80)
    print(f"CF3 Feature Field Normalization (Autoencoder-based)")
    print(f"=" * 80)
    print(f"📂 Input PLY:     {input_path}")
    print(f"📂 Output PLY:    {output_path}")
    print(f"🤖 Autoencoder:   {autoencoder_path}")
    print(f"🔧 Method:        {method}")
    print(f"💻 Device:        {device}")
    print(f"=" * 80)
    
    # Validate inputs
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    if not os.path.exists(autoencoder_path):
        raise FileNotFoundError(f"Autoencoder not found: {autoencoder_path}")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[0/6] Using device: {device}")
    
    # Load autoencoder
    print(f"\n[1/6] Loading autoencoder...")
    checkpoint = torch.load(autoencoder_path, map_location=device)
    
    # Try to infer dimensions from checkpoint
    if 'encoder.0.weight' in checkpoint:
        input_dim = checkpoint['encoder.0.weight'].shape[1]  # Input features
        latent_dim = checkpoint['encoder.8.weight'].shape[0]  # Last encoder layer output (latent dim)
    else:
        # Default CF3 dimensions
        input_dim = 512
        latent_dim = 3
        print(f"⚠️  Could not infer dimensions from checkpoint, using defaults")
    
    print(f"  Input dim:  {input_dim} (CLIP feature dimension)")
    print(f"  Latent dim: {latent_dim} (Compressed dimension)")
    
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    print(f"✓ Autoencoder loaded successfully")
    
    # Load PLY
    print(f"\n[2/6] Loading PLY file...")
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    num_points = len(vertex['x'])
    print(f"✓ Loaded {num_points:,} points")
    
    # Identify compressed feature columns
    print(f"\n[3/6] Identifying compressed feature columns...")
    all_props = [p.name for p in vertex.properties]
    
    # CF3 stores compressed features as f_dc_0, f_dc_1, f_dc_2 (reusing color channels)
    # These are NOT the original RGB colors but the compressed CLIP features!
    feature_props = []
    
    # Check if this is a CF3 feature field (has exactly 3 f_dc channels)
    f_dc_props = [p for p in all_props if p.startswith('f_dc_')]
    
    if len(f_dc_props) == 3:
        # This is CF3 compact feature field format
        print(f"  Detected CF3 format: f_dc_0, f_dc_1, f_dc_2 are compressed features")
        feature_props = sorted(f_dc_props)
    else:
        # Try to find other feature naming conventions (f_0, f_1, feat_0, etc.)
        exclude_keywords = ['dc', 'rest']
        for prop in all_props:
            if prop.startswith(feature_prefix):
                if any(keyword in prop for keyword in exclude_keywords):
                    continue
                feature_props.append(prop)
        
        # Sort by index
        try:
            feature_props.sort(
                key=lambda x: int(x.split('_')[-1]) 
                if '_' in x and x.split('_')[-1].isdigit() 
                else x
            )
        except Exception:
            feature_props.sort()
    
    if not feature_props:
        print(f"\n❌ Error: No CF3 features found!")
        print(f"   Expected: f_0, f_1, f_2 (compressed features)")
        print(f"\n📋 Available properties:")
        for prop in all_props[:20]:
            print(f"   - {prop}")
        raise ValueError("No CF3 compressed features found in PLY file")
    
    if len(feature_props) != latent_dim:
        print(f"\n⚠️  Warning: Expected {latent_dim} features but found {len(feature_props)}")
        print(f"   Features: {feature_props}")
    
    print(f"✓ Found {len(feature_props)} compressed feature dimensions: {feature_props}")
    
    # Extract compressed features
    print(f"\n[4/6] Decoding features to high-dimensional space...")
    compressed_features = np.zeros((num_points, len(feature_props)), dtype=np.float32)
    for i, prop in enumerate(feature_props):
        compressed_features[:, i] = vertex[prop]
    
    print(f"  Compressed shape: {compressed_features.shape}")
    
    # Decode to high-dimensional space
    with torch.no_grad():
        compressed_tensor = torch.from_numpy(compressed_features).to(device)
        decoded_features = autoencoder.decode(compressed_tensor)
        decoded_np = decoded_features.cpu().numpy()
    
    print(f"  Decoded shape:    {decoded_np.shape}")
    print(f"  Memory:           {decoded_np.nbytes / (1024**2):.2f} MB")
    
    # Normalize in high-dimensional space
    print(f"\n[5/6] Normalizing in high-dimensional space...")
    
    mean_vec = np.mean(decoded_np, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    std_vec = np.std(decoded_np, axis=0)
    
    print(f"\n📊 Original High-Dim Statistics:")
    print(f"  Mean Vector Norm: {mean_norm:.6f}")
    print(f"  Mean of Means:    {np.mean(mean_vec):.6f}")
    print(f"  Avg Std:          {np.mean(std_vec):.6f}")
    
    # Apply normalization
    decoded_normalized = decoded_np - mean_vec
    
    if method == "whitening":
        epsilon = 1e-8
        decoded_normalized = decoded_normalized / (std_vec + epsilon)
        print(f"\n✓ Applied: Mean Subtraction + Variance Scaling (Whitening)")
    else:
        print(f"\n✓ Applied: Mean Subtraction (Centering)")
    
    # Verify normalization
    new_mean_norm = np.linalg.norm(np.mean(decoded_normalized, axis=0))
    print(f"\n📊 Normalized High-Dim Statistics:")
    print(f"  Mean Vector Norm: {new_mean_norm:.6f} (should be ~0)")
    
    # Re-encode normalized features
    print(f"\n[6/6] Re-encoding to compressed space...")
    with torch.no_grad():
        normalized_tensor = torch.from_numpy(decoded_normalized.astype(np.float32)).to(device)
        reencoded_features = autoencoder.encode(normalized_tensor)
        reencoded_np = reencoded_features.cpu().numpy()
    
    print(f"  Re-encoded shape: {reencoded_np.shape}")
    
    # Update PLY data
    print(f"\n💾 Building new PLY structure...")
    dtype_list = [(p.name, vertex[p.name].dtype) for p in vertex.properties]
    new_vertex_data = np.zeros(num_points, dtype=dtype_list)
    
    # Copy non-feature properties
    for prop in all_props:
        if prop not in feature_props:
            new_vertex_data[prop] = vertex[prop]
    
    # Write normalized compressed features
    for i, prop in enumerate(feature_props):
        new_vertex_data[prop] = reencoded_np[:, i]
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([el]).write(output_path)
    
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"✓ Saved successfully ({file_size:.2f} MB)")
    
    print(f"\n" + "=" * 80)
    print(f"✅ CF3 Feature Normalization Complete!")
    print(f"=" * 80)
    print(f"\n💡 What happened:")
    print(f"  1. Decoded f_0,f_1,f_2 → {input_dim}-dim CLIP space")
    print(f"  2. Removed dominant 'object/car' signal (mean norm: {mean_norm:.4f} → {new_mean_norm:.6f})")
    print(f"  3. Re-encoded to compressed 3D space")
    print(f"\n📌 Expected improvements:")
    print(f"  - Better discrimination of car parts (wheel, door, trunk, etc.)")
    print(f"  - Reduced false positives from generic 'object/car' queries")
    print(f"  - Higher precision for specific part queries")
    print(f"=" * 80 + "\n")




def main():
    parser = argparse.ArgumentParser(
        description="Normalize CF3 Feature Field with Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic normalization (recommended)
  python feature_field_normalization.py \\
      -i 3d_simulator/source/toycar/point_cloud.ply \\
      -o 3d_simulator/source/toycar/point_cloud_normalized.ply \\
      -a 3d_simulator/source/toycar/autoencoder.pth
  
  # With whitening (stronger normalization)
  python feature_field_normalization.py \\
      -i 3d_simulator/source/toycar/point_cloud.ply \\
      -o 3d_simulator/source/toycar/point_cloud_normalized.ply \\
      -a 3d_simulator/source/toycar/autoencoder.pth \\
      --method whitening
  
  # Use CPU if CUDA unavailable
  python feature_field_normalization.py \\
      -i point_cloud.ply \\
      -o point_cloud_normalized.ply \\
      -a autoencoder.pth \\
      --device cpu

How it works:
  1. Loads compressed features (f_0, f_1, f_2) from PLY file
  2. Uses autoencoder decoder to expand to high-dimensional CLIP space (512-dim)
  3. Subtracts mean vector to remove dominant 'object/car' signals
  4. Re-encodes normalized features back to 3D compressed space
  5. Saves normalized PLY file

This improves part-specific queries like 'right front door', 'wheel' by
removing the common 'car/object' component that dominates all features.
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CF3 PLY file with compressed features (f_0, f_1, f_2)"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output normalized PLY path"
    )
    
    parser.add_argument(
        "-a", "--autoencoder",
        required=True,
        help="Path to CF3 autoencoder.pth model file"
    )
    
    parser.add_argument(
        "--prefix",
        default="f_",
        help="Prefix of feature columns (default: 'f_')"
    )
    
    parser.add_argument(
        "--method",
        default="mean_subtraction",
        choices=["mean_subtraction", "whitening"],
        help="Normalization method. "
             "'mean_subtraction' removes dominant signal (recommended). "
             "'whitening' also scales variance."
    )
    
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for autoencoder (default: cuda)"
    )
    
    args = parser.parse_args()
    
    try:
        normalize_ply_features(
            args.input,
            args.output,
            args.autoencoder,
            args.prefix,
            args.method,
            args.device
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

