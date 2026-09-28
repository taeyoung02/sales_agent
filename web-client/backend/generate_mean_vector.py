#!/usr/bin/env python3
"""
Generate mean_vector.pt for a vehicle dataset.

The mean vector represents the "semantic center" of car-related concepts
and is used for semantic centering in heatmap generation to improve
discrimination between different car parts.
"""

import sys
import torch
from pathlib import Path
from typing import List, Optional

# Default car parts list
PARTS_LIST = [
    "front bumper", "rear bumper", "car hood", "car door", "car roof", 
    "fender", "wheel", "headlight", "taillight", "windshield", 
    "rear window", "side window", "windshield wiper", "side mirror", 
    "car door handle", "license plate", "emblem", "grille", "exhaust pipe"
]


def load_clip_model(device: str = "cpu"):
    """Load CLIP model for text encoding"""
    try:
        import open_clip
        
        print(f"Loading CLIP model...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
        return model, tokenizer
        
    except ImportError:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")


def generate_mean_vector(
    output_path: str,
    parts_list: Optional[List[str]] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generate mean vector from car parts list using CLIP.
    
    Args:
        output_path: Path to save the mean vector .pt file
        parts_list: List of car part names (default: PARTS_LIST)
        device: Device to use for computation (default: "cpu")
    
    Returns:
        Mean vector tensor [512]
    """
    if parts_list is None:
        parts_list = PARTS_LIST
    
    print(f"\n{'='*60}")
    print(f"Generating mean vector from {len(parts_list)} car parts")
    print(f"{'='*60}\n")
    
    # Load CLIP model
    model, tokenizer = load_clip_model(device)
    
    # Encode all parts
    print("Encoding car parts:")
    part_features = []
    
    with torch.no_grad():
        for i, part in enumerate(parts_list, 1):
            # Tokenize and encode (no preprocessing - raw part names to match feature field)
            tokens = tokenizer([part]).to(device)
            text_features = model.encode_text(tokens)
            
            # Normalize and convert to float32
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze().to(torch.float32)
            
            part_features.append(text_features)
            print(f"  [{i}/{len(parts_list)}] {part:20s} -> {text_features.shape}")
    
    # Stack and compute mean
    part_features_tensor = torch.stack(part_features)
    mean_vector = part_features_tensor.mean(dim=0)
    
    print(f"\n✅ Mean vector computed")
    print(f"   Shape: {mean_vector.shape}")
    print(f"   Norm: {mean_vector.norm().item():.4f}")
    
    # Save to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(mean_vector, output_path)
    print(f"\n💾 Saved to: {output_path}")
    
    return mean_vector


def main(dataset_name: str = 'test'):
    """Generate mean_vector.pt for a dataset."""
    # Paths
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / 'source' / dataset_name / 'mean_vector.pt'
    
    # Check if already exists
    if output_path.exists():
        print(f"\n⚠️  Mean vector already exists at: {output_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Generate mean vector
    try:
        mean_vector = generate_mean_vector(str(output_path))
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully generated mean_vector.pt")
        print(f"{'='*60}")
        print(f"\nDataset: {dataset_name}")
        print(f"Path: {output_path}")
        print(f"Vector shape: {mean_vector.shape}")
        print(f"\nThis mean vector will be automatically used for semantic")
        print(f"centering in heatmap generation to improve discrimination")
        print(f"between different car parts.")
        
    except Exception as e:
        print(f"\n❌ Error generating mean vector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'test'
    main(dataset_name)
