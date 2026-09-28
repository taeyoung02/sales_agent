import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open_clip

# 확인해볼 파일 경로 (가장 첫번째 파일 추천)
# pt_file_path = "data/kolon_sample_floating_langsplat/langsplat_features/seq_0001_frame57_score231_fmap_CxHxW.pt" # 경로 수정 필요
# pt_file_path = "data/kolon_sample_floor_langsplat/langsplat_features/seq_0003_frame107_score250_fmap_CxHxW.pt"
pt_file_path = "/home/kolon/Desktop/work/Pipeline/data/volvo_floor_langsplat/langsplat_features/KakaoTalk_Photo_2025-11-25-20-58-30 001_fmap_CxHxW.pt" # 경로 수정 필요
# pt_file_path = "data/bugatti_langsplat/langsplat_features/seq_0007_frame218_score117_fmap_CxHxW.pt" # 경로 수정 필요

# Candidate words to interpret feature vectors
CANDIDATE_WORDS = [
    # Vehicle parts
    "car", "vehicle", "automobile", "wheel", "tire", "windshield", "window", "door", 
    "headlight", "taillight", "bumper", "hood", "roof", "mirror", "grille", "fender",
    "chassis", "body", "paint", "metal", "chrome", "glass", "rubber",
    
    # Background/environment
    "background", "road", "asphalt", "pavement", "ground", "floor", "street", "concrete",
    "sky", "cloud", "building", "wall", "fence", "tree", "grass", "nature",
    "environment", "surroundings", "scenery", "outdoor", "indoor",
    
    # Materials & textures
    "smooth", "rough", "shiny", "matte", "glossy", "reflective", "dark", "bright",
    "black", "white", "gray", "red", "blue", "silver", "shadow", "light",
    
    # Objects
    "object", "thing", "item", "part", "component", "detail", "surface", "edge",
    "nothing", "empty", "void", "blank"
]
try:
    # 1. Load Data
    # Shape: [512, H, W]
    feature_tensor = torch.load(pt_file_path, map_location='cpu').float()
    C, H, W = feature_tensor.shape
    print(f"pt_file_path: {pt_file_path}")
    print(f"Original Shape: {feature_tensor.shape}")
    print(f"Total pixels: {H * W}")
    print("=" * 80)

    # 2. Reshape to [H*W, C] - 각 픽셀이 512차원 벡터
    features_flat = feature_tensor.permute(1, 2, 0).reshape(-1, C).numpy()
    print(f"Flattened shape: {features_flat.shape}")
    
    # 3. 빠르게 0벡터 찾기 (메모리 효율적)
    print("\n🔍 Checking for zero vectors...")
    
    # 각 벡터의 norm 계산 (0벡터는 norm이 0)
    norms = np.linalg.norm(features_flat, axis=1)
    zero_vector_mask = np.isclose(norms, 0, atol=1e-6)
    zero_count = np.sum(zero_vector_mask)
    
    print(f"\n📊 Zero Vector Analysis:")
    print(f"  Total zero vectors found: {zero_count}")
    print(f"  Percentage: {(zero_count / len(features_flat)) * 100:.2f}%")
    
    if zero_count > 0:
        print(f"  ⚠️  ZERO VECTORS DETECTED! (likely background pixels)")
        
        # 0벡터의 위치 찾기
        zero_indices = np.where(zero_vector_mask)[0]
        print(f"\n  First 10 zero vector positions (flat index): {zero_indices[:10]}")
        
        # 2D 좌표로 변환
        zero_coords_y = zero_indices // W
        zero_coords_x = zero_indices % W
        print(f"  First 10 zero vector positions (y, x):")
        for i in range(min(10, len(zero_indices))):
            print(f"    Position {i+1}: ({zero_coords_y[i]}, {zero_coords_x[i]})")
    else:
        print(f"  ✅ No zero vectors found!")
    
    # 4. 추가로 매우 작은 norm을 가진 벡터들도 체크 (거의 0인 벡터)
    print(f"\n🔍 Checking for near-zero vectors (norm < 0.01)...")
    near_zero_mask = norms < 0.01
    near_zero_count = np.sum(near_zero_mask)
    print(f"  Near-zero vectors: {near_zero_count} ({(near_zero_count / len(features_flat)) * 100:.2f}%)")
    
    # 5. Norm 분포 확인
    print(f"\n📊 Vector Norm Statistics:")
    print(f"  Min norm: {norms.min():.6f}")
    print(f"  Max norm: {norms.max():.6f}")
    print(f"  Mean norm: {norms.mean():.6f}")
    print(f"  Median norm: {np.median(norms):.6f}")
    print(f"  Std norm: {norms.std():.6f}")
    
    # 6. 가장 작은 norm을 가진 벡터 10개 확인
    print(f"\n🔍 Top 10 smallest norm vectors:")
    smallest_indices = np.argsort(norms)[:10]
    for i, idx in enumerate(smallest_indices):
        y, x = idx // W, idx % W
        print(f"  #{i+1}: norm={norms[idx]:.6f}, pos=({y},{x}), first 5 dims={features_flat[idx][:5]}")
    
    print("\n" + "=" * 80)
    
    # 7. Find most common unique vectors
    print("\n🔍 Finding most common feature vectors...")
    unique_vectors, inverse_indices, counts = np.unique(
        features_flat, 
        axis=0, 
        return_inverse=True, 
        return_counts=True
    )
    
    # Sort by frequency
    sorted_indices = np.argsort(-counts)
    top_5_indices = sorted_indices[:5]
    
    print(f"\n📊 Top 5 Most Common Feature Vectors:")
    print(f"  Total unique vectors: {len(unique_vectors)}")
    print(f"  Total pixels: {len(features_flat)}")
    print()
    
    for rank, idx in enumerate(top_5_indices, 1):
        count = counts[idx]
        percentage = (count / len(features_flat)) * 100
        vector = unique_vectors[idx]
        norm = np.linalg.norm(vector)
        
        print(f"  Rank #{rank}:")
        print(f"    Frequency: {count} pixels ({percentage:.2f}%)")
        print(f"    Norm: {norm:.6f}")
        if np.allclose(vector, 0):
            print(f"    Type: ⚠️  ZERO VECTOR (background)")
        print()
    
    # 8. Initialize OpenCLIP for semantic interpretation
    print("\n" + "=" * 80)
    print("\n🧠 Initializing OpenCLIP for semantic interpretation...")
    print("  Model: ViT-B-32, Pretrained: openai")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained='openai'
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Encode all candidate words
    print(f"  Encoding {len(CANDIDATE_WORDS)} candidate words...")
    with torch.no_grad():
        text_tokens = tokenizer(CANDIDATE_WORDS).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
    
    print("✓ OpenCLIP ready for interpretation")
    
    # 9. Interpret top 5 most common vectors
    print("\n" + "=" * 80)
    print("\n🎯 Semantic Interpretation of Top 5 Most Common Vectors:")
    print()
    
    for rank, idx in enumerate(top_5_indices, 1):
        count = counts[idx]
        percentage = (count / len(features_flat)) * 100
        vector = unique_vectors[idx]
        norm = np.linalg.norm(vector)
        
        # Normalize vector for cosine similarity
        if norm > 0:
            vector_normalized = vector / norm
        else:
            vector_normalized = vector
        
        # Compute cosine similarity with all candidate words
        similarities = text_features @ vector_normalized
        
        # Get top 5 matches
        top_match_indices = np.argsort(-similarities)[:5]
        
        print(f"  📌 Rank #{rank} ({count} pixels, {percentage:.2f}%):")
        if np.allclose(vector, 0):
            print(f"     ⚠️  ZERO VECTOR - No semantic meaning (likely masked/background)")
        else:
            print(f"     Top 5 semantic matches:")
            for i, match_idx in enumerate(top_match_indices, 1):
                word = CANDIDATE_WORDS[match_idx]
                similarity = similarities[match_idx]
                print(f"       {i}. '{word}' (similarity: {similarity:.4f})")
        print()
    
    print("=" * 80)
    
    # 10. PCA 시각화 (512 dim -> 3 dim RGB)
    print("\n🎨 Running PCA for visualization...")
    print("  This might take a moment...")
    
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_flat)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    print(f"\n  PCA Explained Variance:")
    print(f"    PC1: {explained_var[0]*100:.2f}%")
    print(f"    PC2: {explained_var[1]*100:.2f}%")
    print(f"    PC3: {explained_var[2]*100:.2f}%")
    print(f"    Total: {sum(explained_var)*100:.2f}%")
    
    # Normalize to [0, 1] for RGB visualization
    features_pca_norm = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    
    # Reshape back to image: [H, W, 3]
    feature_image = features_pca_norm.reshape(H, W, 3)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA visualization
    axes[0].imshow(feature_image)
    axes[0].set_title(f"PCA Visualization (RGB)\n{explained_var[0]*100:.1f}% + {explained_var[1]*100:.1f}% + {explained_var[2]*100:.1f}% = {sum(explained_var)*100:.1f}% variance", 
                      fontsize=12)
    axes[0].axis('off')
    
    # Zero vector heatmap
    zero_heatmap = zero_vector_mask.reshape(H, W)
    axes[1].imshow(zero_heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"Zero Vector Heatmap\n{zero_count} zero vectors ({(zero_count/len(features_flat))*100:.2f}%)", 
                      fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure to current directory
    import os
    filename = os.path.basename(pt_file_path).replace('.pt', '_analysis.png')
    output_path = filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Also save a simple version for quick preview
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_image)
    plt.title(f"PCA Feature Visualization\n{os.path.basename(pt_file_path)}", fontsize=10)
    plt.axis('off')
    simple_output_path = "debug_feature_vis.png"
    plt.savefig(simple_output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Simple visualization saved: {simple_output_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"\nSummary:")
    print(f"  - Zero vectors: {zero_count}/{len(features_flat)} ({(zero_count/len(features_flat))*100:.2f}%)")
    print(f"  - Mean norm: {norms.mean():.6f}")
    print(f"  - PCA variance captured: {sum(explained_var)*100:.2f}%")
    print(f"  - Output files:")
    print(f"    • {output_path}")
    print(f"    • {simple_output_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")