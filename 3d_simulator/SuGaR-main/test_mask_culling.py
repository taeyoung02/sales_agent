#!/usr/bin/env python3
"""
Quick Test Script for Mask-based Voting Culling
================================================

This script helps you validate that the mask-based culling implementation
is working correctly before running a full mesh extraction.

Usage:
    python test_mask_culling.py --scene_path data/volvo --mask_dir data/volvo/masks
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add SuGaR modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sugar_utils.mask_voting_culling import load_binary_masks
from rich.console import Console

CONSOLE = Console(width=120)


class MockCamera:
    """Mock camera for testing"""
    def __init__(self, image_name):
        self.image_name = image_name


def check_mask_directory(mask_dir):
    """Check if mask directory exists and contains valid masks"""
    CONSOLE.print(f"\n[bold cyan]Checking mask directory: {mask_dir}[/bold cyan]")
    
    mask_path = Path(mask_dir)
    
    if not mask_path.exists():
        CONSOLE.print(f"[red]✗ Directory does not exist: {mask_dir}[/red]")
        return False
    
    # Find all mask files
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
        mask_files.extend(list(mask_path.glob(ext)))
    
    if len(mask_files) == 0:
        CONSOLE.print(f"[red]✗ No mask files found in {mask_dir}[/red]")
        return False
    
    CONSOLE.print(f"[green]✓ Found {len(mask_files)} mask files[/green]")
    
    # Check first few masks
    CONSOLE.print("\nValidating first 3 masks:")
    for i, mask_file in enumerate(mask_files[:3]):
        try:
            mask = Image.open(mask_file).convert('L')
            mask_np = np.array(mask)
            
            unique_values = np.unique(mask_np)
            fg_ratio = (mask_np > 127).sum() / mask_np.size * 100
            
            CONSOLE.print(f"  {i+1}. {mask_file.name}")
            CONSOLE.print(f"     Size: {mask_np.shape}")
            CONSOLE.print(f"     Unique values: {unique_values[:10]}...")  # Show first 10
            CONSOLE.print(f"     Foreground ratio: {fg_ratio:.1f}%")
            
            if fg_ratio < 1.0:
                CONSOLE.print(f"     [yellow]⚠ Warning: Very small foreground (<1%)[/yellow]")
            elif fg_ratio > 99.0:
                CONSOLE.print(f"     [yellow]⚠ Warning: Very large foreground (>99%)[/yellow]")
            else:
                CONSOLE.print(f"     [green]✓ Looks good[/green]")
                
        except Exception as e:
            CONSOLE.print(f"  {i+1}. {mask_file.name}")
            CONSOLE.print(f"     [red]✗ Error: {e}[/red]")
    
    return True


def test_mask_loading(scene_path, mask_dir):
    """Test the mask loading functionality"""
    CONSOLE.print(f"\n[bold cyan]Testing mask loading functionality[/bold cyan]")
    
    # Find camera names from scene images
    image_dir = Path(scene_path) / 'images'
    if not image_dir.exists():
        CONSOLE.print(f"[red]✗ Image directory not found: {image_dir}[/red]")
        return False
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
        image_files.extend(list(image_dir.glob(ext)))
    
    if len(image_files) == 0:
        CONSOLE.print(f"[red]✗ No images found in {image_dir}[/red]")
        return False
    
    CONSOLE.print(f"Found {len(image_files)} images in scene")
    
    # Create mock cameras
    cameras = []
    for img_file in image_files[:10]:  # Test with first 10
        cam_name = img_file.stem  # Filename without extension
        cameras.append(MockCamera(cam_name))
    
    CONSOLE.print(f"Testing with first {len(cameras)} cameras")
    
    # Try to load masks
    try:
        masks_dict = load_binary_masks(
            cameras=cameras,
            mask_dir=mask_dir,
            dilate_kernel_size=7,
            device='cpu'  # Use CPU for testing
        )
        
        CONSOLE.print(f"\n[bold green]✓ Successfully loaded {len(masks_dict)} masks[/bold green]")
        
        if len(masks_dict) < len(cameras) * 0.5:
            CONSOLE.print(f"[yellow]⚠ Warning: Only {len(masks_dict)}/{len(cameras)} masks loaded[/yellow]")
            CONSOLE.print("[yellow]  This may indicate filename mismatch[/yellow]")
        
        # Show statistics for first mask
        if len(masks_dict) > 0:
            first_cam = list(masks_dict.keys())[0]
            first_mask = masks_dict[first_cam]
            
            CONSOLE.print(f"\nFirst mask statistics (camera: {first_cam}):")
            CONSOLE.print(f"  Shape: {first_mask.shape}")
            CONSOLE.print(f"  Device: {first_mask.device}")
            CONSOLE.print(f"  Dtype: {first_mask.dtype}")
            CONSOLE.print(f"  Min/Max: {first_mask.min():.3f} / {first_mask.max():.3f}")
            CONSOLE.print(f"  Foreground ratio: {(first_mask > 0.5).float().mean() * 100:.1f}%")
        
        return True
        
    except Exception as e:
        CONSOLE.print(f"[red]✗ Error loading masks: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def estimate_culling_impact(scene_path, mask_dir):
    """Estimate how many points will be culled"""
    CONSOLE.print(f"\n[bold cyan]Estimating culling impact[/bold cyan]")
    
    # This is a simplified estimation
    # In reality, we'd need to load the actual Gaussians and project them
    
    image_dir = Path(scene_path) / 'images'
    mask_path = Path(mask_dir)
    
    if not image_dir.exists() or not mask_path.exists():
        CONSOLE.print("[yellow]⚠ Cannot estimate impact - directories missing[/yellow]")
        return
    
    # Sample a few masks
    mask_files = list(mask_path.glob('*.png')) + list(mask_path.glob('*.jpg'))
    
    if len(mask_files) == 0:
        return
    
    # Calculate average foreground ratio
    fg_ratios = []
    for mask_file in mask_files[:20]:  # Sample 20 masks
        try:
            mask = Image.open(mask_file).convert('L')
            mask_np = np.array(mask)
            fg_ratio = (mask_np > 127).sum() / mask_np.size
            fg_ratios.append(fg_ratio)
        except:
            pass
    
    if len(fg_ratios) > 0:
        avg_fg = np.mean(fg_ratios) * 100
        avg_bg = 100 - avg_fg
        
        CONSOLE.print(f"\nAverage mask statistics (from {len(fg_ratios)} masks):")
        CONSOLE.print(f"  Foreground (object): {avg_fg:.1f}%")
        CONSOLE.print(f"  Background: {avg_bg:.1f}%")
        CONSOLE.print(f"\n[bold]Estimated Gaussian removal:[/bold]")
        CONSOLE.print(f"  With threshold=0.75: ~{avg_bg * 0.6:.1f}% - {avg_bg * 0.8:.1f}%")
        CONSOLE.print(f"  With threshold=0.85: ~{avg_bg * 0.5:.1f}% - {avg_bg * 0.7:.1f}%")
        
        if avg_bg > 70:
            CONSOLE.print(f"\n[yellow]⚠ Large background detected ({avg_bg:.1f}%)[/yellow]")
            CONSOLE.print("[yellow]  Consider using aggressive threshold (0.6-0.7)[/yellow]")
        elif avg_bg < 30:
            CONSOLE.print(f"\n[green]✓ Small background ({avg_bg:.1f}%)[/green]")
            CONSOLE.print("[green]  Conservative threshold (0.8-0.9) recommended[/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Test mask-based voting culling setup'
    )
    parser.add_argument('--scene_path', type=str, required=True,
                        help='Path to scene data')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to mask directory')
    
    args = parser.parse_args()
    
    CONSOLE.print("\n" + "="*80)
    CONSOLE.print("[bold cyan]Mask-based Voting Culling - Setup Validation[/bold cyan]")
    CONSOLE.print("="*80)
    
    # Run tests
    all_good = True
    
    # 1. Check mask directory
    if not check_mask_directory(args.mask_dir):
        all_good = False
    
    # 2. Test mask loading
    if not test_mask_loading(args.scene_path, args.mask_dir):
        all_good = False
    
    # 3. Estimate impact
    estimate_culling_impact(args.scene_path, args.mask_dir)
    
    # Summary
    CONSOLE.print("\n" + "="*80)
    if all_good:
        CONSOLE.print("[bold green]✓ All checks passed! Ready to extract mesh.[/bold green]")
        CONSOLE.print("\nRecommended command:")
        CONSOLE.print(f"[cyan]python extract_mesh.py \\")
        CONSOLE.print(f"    -s {args.scene_path} \\")
        CONSOLE.print(f"    -c <checkpoint_path> \\")
        CONSOLE.print(f"    -m <sugar_model_path> \\")
        CONSOLE.print(f"    --mask_dir {args.mask_dir} \\")
        CONSOLE.print(f"    --mask_voting_threshold 0.75 \\")
        CONSOLE.print(f"    --mask_dilation_kernel 7 \\")
        CONSOLE.print(f"    --debug_culling True[/cyan]")
    else:
        CONSOLE.print("[bold red]✗ Some checks failed. Please fix the issues above.[/bold red]")
    CONSOLE.print("="*80 + "\n")
    
    return 0 if all_good else 1


if __name__ == '__main__':
    sys.exit(main())
