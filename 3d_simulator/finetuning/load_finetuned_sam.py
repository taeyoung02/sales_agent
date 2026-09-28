"""
Utility to load fine-tuned SAM model for use in CLIP training
"""
import torch
from segment_anything import sam_model_registry


def load_finetuned_sam(
    checkpoint_path: str,
    sam_model_type: str = "vit_b",
    device: str = "cuda"
):
    """
    Load a fine-tuned SAM model checkpoint
    
    Args:
        checkpoint_path: Path to fine-tuned SAM checkpoint (best_model.pth or checkpoint_epoch_X.pth)
        sam_model_type: SAM model type (vit_b, vit_l, vit_h)
        device: Device to load model on
        
    Returns:
        SAM model with fine-tuned weights loaded
    """
    print(f"Loading fine-tuned SAM from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check checkpoint format
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        val_iou = checkpoint.get('val_iou', 'unknown')
        print(f"  Checkpoint info:")
        print(f"    - Epoch: {epoch}")
        print(f"    - Validation IoU: {val_iou}")
    else:
        # Direct state dict
        state_dict = checkpoint
    
    # Create SAM model (without loading pretrained weights)
    # We need to initialize the model structure first
    print(f"  Creating SAM model structure ({sam_model_type})...")
    sam = sam_model_registry[sam_model_type](checkpoint=None)
    
    # Load fine-tuned weights
    print(f"  Loading fine-tuned weights...")
    sam.load_state_dict(state_dict, strict=True)
    
    sam.to(device)
    sam.eval()
    
    print(f"✓ Fine-tuned SAM loaded successfully")
    
    return sam


def get_sam_checkpoint_path(
    finetuning_output_dir: str,
    use_best: bool = True,
    epoch: int = None
):
    """
    Helper to get SAM checkpoint path from finetuning output directory
    
    Args:
        finetuning_output_dir: Path to SAM finetuning output dir (e.g., outputs/sam_20260101_003032)
        use_best: If True, use best_model.pth, otherwise use final_model.pth or specific epoch
        epoch: If specified, load checkpoint from specific epoch
        
    Returns:
        Full path to checkpoint file
    """
    import os
    
    if epoch is not None:
        checkpoint_path = os.path.join(finetuning_output_dir, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif use_best:
        checkpoint_path = os.path.join(finetuning_output_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Best model not found: {checkpoint_path}\n"
                f"Make sure SAM fine-tuning has completed at least one validation epoch"
            )
    else:
        checkpoint_path = os.path.join(finetuning_output_dir, 'final_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Final model not found: {checkpoint_path}")
    
    return checkpoint_path


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_finetuned_sam.py <checkpoint_path>")
        print("Example: python load_finetuned_sam.py outputs/sam_20260101_003032/best_model.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sam = load_finetuned_sam(checkpoint_path, device=device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in sam.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Device: {next(sam.parameters()).device}")
