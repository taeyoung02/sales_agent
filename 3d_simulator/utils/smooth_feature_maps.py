#!/usr/bin/env python3
"""
Feature Map Smoothing for LangSplat Output

Problem:
DINO creates rigid, steep boundaries → SAM uses them for masking → 
Feature vectors have large distance gaps → Gradients diverge in CF3 → NaN values

Solution:
Apply Gaussian blur to smooth feature maps, reducing gradient magnitude 
and preventing divergence during CF3 training.

Usage:
    python smooth_feature_maps.py --feature_dir /workspace/data/PROJECT_langsplat/language_features
"""

import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path


def reconstruct_spatial_feature_map(features, seg_map):
    """
    Reconstruct spatial feature map from flattened features and segmentation map.
    
    LangSplat saves features as (N_masks, 512) - flattened by mask index.
    We need to reconstruct (H, W, 512) using segmentation map indices.
    
    Args:
        features: Flattened feature array (N_masks, 512)
        seg_map: Segmentation map (4, H, W) with mask indices (-1 for background)
    
    Returns:
        Spatial feature map (H, W, 512) or None if reconstruction fails
    """
    if len(features.shape) != 2:
        print(f"  Error: Expected features shape (N, 512), got {features.shape}")
        return None
    
    if len(seg_map.shape) != 3:
        print(f"  Error: Expected seg_map shape (4, H, W), got {seg_map.shape}")
        return None
    
    n_masks, feature_dim = features.shape
    _, H, W = seg_map.shape
    
    # Use the first segmentation level (default level)
    # seg_map[0] contains indices 0, 1, 2, ... (n_masks-1), and -1 for background
    seg_indices = seg_map[0]  # (H, W)
    
    # Initialize spatial feature map (zeros for background)
    spatial_map = np.zeros((H, W, feature_dim), dtype=np.float32)
    
    # Fill in features according to segmentation indices
    for mask_idx in range(n_masks):
        mask_pixels = (seg_indices == mask_idx)
        if np.sum(mask_pixels) > 0:
            # Broadcast feature vector to all pixels belonging to this mask
            spatial_map[mask_pixels] = features[mask_idx]
    
    return spatial_map


def flatten_spatial_feature_map(spatial_map, seg_map, original_n_masks):
    """
    Flatten spatial feature map back to (N_masks, 512) format.
    
    CRITICAL: Preserve original mask count to avoid shape mismatch.
    
    Args:
        spatial_map: Spatial feature map (H, W, 512)
        seg_map: Segmentation map (4, H, W) with mask indices
        original_n_masks: Original number of masks (must match output shape)
    
    Returns:
        Flattened features (N_masks, 512) with preserved mask count
    """
    H, W, feature_dim = spatial_map.shape
    seg_indices = seg_map[0]  # (H, W)
    
    # Initialize output with original mask count
    flattened_features = np.zeros((original_n_masks, feature_dim), dtype=np.float32)
    
    # Average features within each mask region (using original mask indices 0 to N-1)
    for mask_idx in range(original_n_masks):
        mask_pixels = (seg_indices == mask_idx)
        pixel_count = np.sum(mask_pixels)
        
        if pixel_count > 0:
            # Take mean of all pixel features in this mask
            flattened_features[mask_idx] = np.mean(spatial_map[mask_pixels], axis=0)
        else:
            # Mask disappeared after smoothing (very small mask absorbed by neighbors)
            # Keep zero vector (will be ignored by CF3 anyway)
            # Alternative: copy from nearest neighbor mask
            pass
    
    return flattened_features


def smooth_feature_map(feature_map, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to spatial feature map.
    
    Args:
        feature_map: Spatial feature map (H, W, C)
        kernel_size: Gaussian kernel size (default: (5, 5))
        sigma: Gaussian sigma (0 = auto-calculate)
    
    Returns:
        Smoothed feature map with same shape
    """
    if len(feature_map.shape) != 3:
        print(f"  Error: Expected spatial map shape (H, W, C), got {feature_map.shape}")
        return None
    
    # Apply Gaussian blur (cv2 requires float32)
    smoothed = cv2.GaussianBlur(
        feature_map.astype(np.float32), 
        kernel_size, 
        sigma
    )
    
    return smoothed


def process_feature_files(feature_dir, kernel_size=(5, 5), sigma=0, dry_run=False, backup=True):
    """
    Process all *_f.npy files in feature directory.
    
    Args:
        feature_dir: Directory containing *_f.npy and *_s.npy files
        kernel_size: Gaussian kernel size
        sigma: Gaussian sigma
        dry_run: If True, only analyze without modifying files
        backup: If True, backup original files as *_original.npy before overwriting
    """
    # Find all feature files (*_f.npy = feature embeddings)
    feature_files = glob.glob(os.path.join(feature_dir, "*_f.npy"))
    
    if len(feature_files) == 0:
        print(f"❌ No *_f.npy files found in {feature_dir}")
        print(f"   Make sure LangSplat preprocessing completed successfully")
        return
    
    print(f"\n{'='*70}")
    print(f"Feature Map Smoothing with Spatial Reconstruction")
    print(f"{'='*70}")
    print(f"  Directory: {feature_dir}")
    print(f"  Files found: {len(feature_files)}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Sigma: {sigma} (auto)" if sigma == 0 else f"  Sigma: {sigma}")
    print(f"  Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print(f"{'='*70}\n")
    
    # Statistics tracking
    stats = {
        'total': len(feature_files),
        'smoothed': 0,
        'skipped': 0,
        'errors': 0,
        'shapes': {},
        'reconstructed_shapes': {}
    }
    
    for fpath in tqdm(feature_files, desc="Smoothing feature maps"):
        try:
            # Load feature map and corresponding segmentation map
            seg_path = fpath.replace('_f.npy', '_s.npy')
            
            if not os.path.exists(seg_path):
                print(f"\n  ⚠️  Segmentation file not found: {os.path.basename(seg_path)}")
                print(f"     Skipping {os.path.basename(fpath)}")
                stats['skipped'] += 1
                continue
            
            features = np.load(fpath)  # (N_masks, 512)
            seg_map = np.load(seg_path)  # (4, H, W)
            
            original_shape = features.shape
            
            # Track shape distribution
            shape_key = str(original_shape)
            stats['shapes'][shape_key] = stats['shapes'].get(shape_key, 0) + 1
            
            # Step 1: Reconstruct spatial feature map
            spatial_map = reconstruct_spatial_feature_map(features, seg_map)
            
            if spatial_map is None:
                stats['errors'] += 1
                continue
            
            # Track reconstructed shapes
            spatial_shape_key = str(spatial_map.shape)
            stats['reconstructed_shapes'][spatial_shape_key] = \
                stats['reconstructed_shapes'].get(spatial_shape_key, 0) + 1
            
            # Step 2: Apply Gaussian blur to spatial map
            smoothed_spatial = smooth_feature_map(spatial_map, kernel_size, sigma)
            
            if smoothed_spatial is None:
                stats['errors'] += 1
                continue
            
            # Step 3: Flatten back to original format
            smoothed_features = flatten_spatial_feature_map(
                smoothed_spatial, 
                seg_map, 
                original_n_masks=original_shape[0]  # Preserve original mask count
            )
            
            # Verify shape preservation
            if smoothed_features.shape != original_shape:
                print(f"\n  ⚠️  Shape mismatch for {os.path.basename(fpath)}")
                print(f"     Original: {original_shape}, Smoothed: {smoothed_features.shape}")
                stats['errors'] += 1
                continue
            
            # Calculate smoothing impact
            diff = np.abs(smoothed_features - features)
            mean_change = np.mean(diff)
            max_change = np.max(diff)
            
            if not dry_run:
                # Backup original file (first time only) if backup flag is enabled
                if backup:
                    backup_path = fpath.replace('.npy', '_original.npy')
                    if not os.path.exists(backup_path):
                        np.save(backup_path, features)
                
                # Overwrite with smoothed version
                np.save(fpath, smoothed_features)
            
            stats['smoothed'] += 1
            
            # Log first file details
            if stats['smoothed'] == 1:
                print(f"\n  Sample file: {os.path.basename(fpath)}")
                print(f"    Flattened shape: {original_shape}")
                print(f"    Reconstructed shape: {spatial_map.shape}")
                print(f"    Mean change: {mean_change:.6f}")
                print(f"    Max change: {max_change:.6f}")
                print(f"    Value range: [{np.min(smoothed_features):.4f}, {np.max(smoothed_features):.4f}]")
        
        except Exception as e:
            print(f"\n  ❌ Error processing {os.path.basename(fpath)}: {e}")
            import traceback
            traceback.print_exc()
            stats['errors'] += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Smoothing Summary")
    print(f"{'='*70}")
    print(f"  Total files: {stats['total']}")
    print(f"  ✓ Smoothed: {stats['smoothed']}")
    print(f"  ⊘ Skipped: {stats['skipped']}")
    print(f"  ✗ Errors: {stats['errors']}")
    print(f"\n  Original flattened shapes:")
    for shape, count in stats['shapes'].items():
        print(f"    {shape}: {count} files")
    
    if stats['reconstructed_shapes']:
        print(f"\n  Reconstructed spatial shapes:")
        for shape, count in stats['reconstructed_shapes'].items():
            print(f"    {shape}: {count} files")
    
    if not dry_run and stats['smoothed'] > 0:
        if backup:
            print(f"\n  💾 Original files backed up as *_original.npy")
        print(f"  ✓ Smoothed files saved (overwritten)")
    
    print(f"{'='*70}\n")
    
    if dry_run:
        print("  ℹ️  This was a dry run. Re-run without --dry_run to apply changes.")
    else:
        print("  ✓ Smoothing completed! You can now run CF3 training.")


def verify_smoothing(feature_dir):
    """
    Verify smoothing by comparing original and smoothed files.
    """
    feature_files = glob.glob(os.path.join(feature_dir, "*_f.npy"))
    original_files = glob.glob(os.path.join(feature_dir, "*_original.npy"))
    
    if len(original_files) == 0:
        print("  No *_original.npy files found. Cannot verify.")
        return
    
    print(f"\n{'='*70}")
    print(f"Smoothing Verification")
    print(f"{'='*70}")
    
    for orig_path in original_files[:3]:  # Check first 3 files
        smooth_path = orig_path.replace('_original.npy', '.npy')
        
        if not os.path.exists(smooth_path):
            continue
        
        orig_data = np.load(orig_path)
        smooth_data = np.load(smooth_path)
        
        diff = np.abs(smooth_data - orig_data)
        
        print(f"\n  File: {os.path.basename(orig_path)}")
        print(f"    Original range: [{np.min(orig_data):.4f}, {np.max(orig_data):.4f}]")
        print(f"    Smoothed range: [{np.min(smooth_data):.4f}, {np.max(smooth_data):.4f}]")
        print(f"    Mean difference: {np.mean(diff):.6f}")
        print(f"    Max difference: {np.max(diff):.6f}")
        print(f"    % pixels changed > 0.01: {np.sum(diff > 0.01) / diff.size * 100:.2f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smooth LangSplat feature maps to prevent NaN in CF3 training'
    )
    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help='Path to language_features directory (e.g., /workspace/data/PROJECT_langsplat/language_features)'
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        nargs=2,
        default=[5, 5],
        help='Gaussian kernel size (height width). Default: 5 5. Larger = more smoothing. Recommended: 5-9'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0,
        help='Gaussian sigma. Default: 0 (auto-calculate). Larger = more smoothing.'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Analyze files without modifying them'
    )
    parser.add_argument(
        '--no_backup',
        action='store_true',
        help='Do not backup original files (overwrite directly without creating *_original.npy)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify smoothing by comparing original and smoothed files'
    )
    
    args = parser.parse_args()
    
    # Validate feature directory
    if not os.path.isdir(args.feature_dir):
        print(f"❌ Error: Feature directory not found: {args.feature_dir}")
        exit(1)
    
    # Validate kernel size (must be odd)
    kernel_size = tuple(args.kernel_size)
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        print(f"❌ Error: Kernel size must be odd numbers (got {kernel_size})")
        print(f"   Try: --kernel_size 5 5 or --kernel_size 7 7")
        exit(1)
    
    # Run verification or smoothing
    if args.verify:
        verify_smoothing(args.feature_dir)
    else:
        process_feature_files(
            args.feature_dir,
            kernel_size=kernel_size,
            sigma=args.sigma,
            dry_run=args.dry_run,
            backup=not args.no_backup  # Invert flag: backup=True by default
        )
