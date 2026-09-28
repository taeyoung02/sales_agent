"""
Mask-based Voting Culling for SuGaR
====================================

This module implements a mask-based voting culling algorithm to remove background Gaussians
before mesh extraction. The algorithm projects all Gaussians onto each camera view and
votes whether they appear in the background (outside the binary mask) or foreground.

Gaussians that are consistently voted as background across multiple views are removed.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Union
from pathlib import Path
from rich.console import Console

CONSOLE = Console(width=120)


def load_binary_masks(
    cameras,
    mask_dir: str,
    dilate_kernel_size: int = 5,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Load binary masks for all camera views.
    
    Args:
        cameras: Camera list/wrapper containing all views
        mask_dir: Directory containing binary mask images
        dilate_kernel_size: Kernel size for morphological dilation (0 to disable)
        device: Device to load masks to
        
    Returns:
        Dictionary mapping camera image_name to binary mask tensor (1, H, W)
    """
    masks_dict = {}
    mask_path = Path(mask_dir)
    
    if not mask_path.exists():
        CONSOLE.print(f"[yellow]Warning: Mask directory not found: {mask_dir}[/yellow]")
        return masks_dict
    
    CONSOLE.print(f"Loading masks from: {mask_dir}")
    CONSOLE.print(f"Dilation kernel size: {dilate_kernel_size}")
    
    # Collect all cameras
    if hasattr(cameras, '__len__'):
        cam_list = cameras
    else:
        cam_list = cameras.cameras if hasattr(cameras, 'cameras') else []
    
    loaded_count = 0
    for cam in cam_list:
        cam_name = cam.image_name
        
        # Try different mask file extensions
        mask_file = None
        for ext in ['.png', '.jpg', '.PNG', '.JPG']:
            potential_path = mask_path / f"{cam_name}{ext}"
            if potential_path.exists():
                mask_file = potential_path
                break
        
        if mask_file is None:
            CONSOLE.print(f"[yellow]Warning: Mask not found for camera: {cam_name}[/yellow]")
            continue
        
        # Load mask as grayscale
        mask_img = Image.open(mask_file).convert('L')
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        
        # Apply dilation to expand the foreground region
        if dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (dilate_kernel_size, dilate_kernel_size)
            )
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        
        # Threshold to binary (1=foreground, 0=background)
        mask_np = (mask_np > 0.5).astype(np.float32)
        
        # Convert to tensor [1, H, W]
        mask_tensor = torch.from_numpy(mask_np[None]).to(device)
        masks_dict[cam_name] = mask_tensor
        loaded_count += 1
    
    CONSOLE.print(f"Successfully loaded {loaded_count}/{len(cam_list)} masks")
    return masks_dict


def cull_gaussians_by_mask_voting(
    sugar_model,
    cameras,
    masks_dict: Dict[str, torch.Tensor],
    threshold: float = 0.7,
    device: str = "cuda",
    debug_output_dir: Optional[str] = None,
) -> None:
    """
    Remove background Gaussians using mask-based voting across camera views.
    
    This function modifies the sugar_model in-place by removing Gaussians that are
    consistently projected outside the binary masks across multiple camera views.
    
    Args:
        sugar_model: SuGaR model object with _points, _scales, _quaternions, etc.
        cameras: Camera wrapper or list containing all training views
        masks_dict: Dictionary mapping camera image_name to binary mask tensor (1, H, W)
                   Values: 1=foreground/object, 0=background
        threshold: Background voting ratio threshold (0.0 ~ 1.0)
                  Higher values are more conservative (only remove obvious background)
                  Recommended: 0.7-0.8
        device: Computation device
        debug_output_dir: Optional directory to save debug visualizations
        
    Returns:
        None (modifies sugar_model in-place)
    """
    
    if len(masks_dict) == 0:
        CONSOLE.print("[yellow]No masks loaded. Skipping mask-based voting culling.[/yellow]")
        return
    
    CONSOLE.print("\n" + "="*80)
    CONSOLE.print("[bold cyan]Starting Mask-based Voting Culling[/bold cyan]")
    CONSOLE.print("="*80)
    
    # Get Gaussian centers
    points = sugar_model.points.detach()  # (N, 3)
    n_points = points.shape[0]
    
    CONSOLE.print(f"Initial number of Gaussians: {n_points:,}")
    CONSOLE.print(f"Voting threshold: {threshold:.2f}")
    CONSOLE.print(f"Number of cameras with masks: {len(masks_dict)}")
    
    # Initialize voting tensors
    background_votes = torch.zeros(n_points, device=device)
    visible_counts = torch.zeros(n_points, device=device)
    
    # Collect cameras
    if hasattr(cameras, '__len__'):
        cam_list = cameras
    else:
        cam_list = cameras.cameras if hasattr(cameras, 'cameras') else []
    
    # Vote across all camera views
    CONSOLE.print("\nProjecting Gaussians to camera views...")
    processed_views = 0
    
    for cam_idx, cam in enumerate(cam_list):
        cam_name = cam.image_name
        
        # Skip if mask not available
        if cam_name not in masks_dict:
            continue
        
        processed_views += 1
        
        # Get mask
        mask = masks_dict[cam_name]  # (1, H, W)
        H, W = mask.shape[-2], mask.shape[-1]
        
        # Get camera matrices
        # world_view_transform: World -> Camera
        # full_proj_transform: World -> Clip Space
        view_matrix = cam.world_view_transform.to(device)
        proj_matrix = cam.full_proj_transform.to(device)
        
        # Project points to clip space
        # Convert to homogeneous coordinates
        points_hom = torch.cat([points, torch.ones(n_points, 1, device=device)], dim=1)  # (N, 4)
        
        # Apply projection
        p_hom = points_hom @ proj_matrix.T  # (N, 4)
        p_w = p_hom[:, 3:4]  # (N, 1)
        
        # Filter points behind camera (w <= 0)
        valid_z = p_w[:, 0] > 0.001
        
        # Perspective division to get NDC coordinates (-1 to 1)
        p_ndc = p_hom[:, :3] / (p_w + 1e-7)  # (N, 3)
        
        # Convert NDC to pixel coordinates
        # NDC [-1, 1] -> UV [0, W] and [0, H]
        u = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        v = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        
        # Check if points are within image bounds
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_z
        
        # Get valid indices
        valid_indices = torch.where(in_frame)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Sample mask at projected locations
        u_valid = u[valid_indices].long().clamp(0, W - 1)
        v_valid = v[valid_indices].long().clamp(0, H - 1)
        
        # Sample mask (v=row, u=col)
        mask_values = mask[0, v_valid, u_valid]  # (num_valid,)
        
        # Vote: mask_values < 0.5 means background
        is_background = (mask_values < 0.5).float()
        
        # Update votes
        background_votes[valid_indices] += is_background
        visible_counts[valid_indices] += 1.0
        
        # Debug output for first few views
        if debug_output_dir and cam_idx < 5:
            debug_dir = Path(debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Visualize projection
            debug_img = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            
            # Draw projected points
            for idx in valid_indices[:1000]:  # Limit to 1000 points for visualization
                uu = int(u[idx].item())
                vv = int(v[idx].item())
                if mask_values[valid_indices == idx].item() < 0.5:
                    color = (0, 0, 255)  # Red for background
                else:
                    color = (0, 255, 0)  # Green for foreground
                cv2.circle(debug_img, (uu, vv), 1, color, -1)
            
            cv2.imwrite(str(debug_dir / f"projection_{cam_idx:03d}_{cam_name}.jpg"), debug_img)
        
        if processed_views % 10 == 0:
            CONSOLE.print(f"  Processed {processed_views} views...")
    
    CONSOLE.print(f"Completed projection for {processed_views} views\n")
    
    # Compute voting ratios
    # Points never visible are kept (set visible_counts to 1 to avoid division by zero)
    visible_counts[visible_counts == 0] = 1.0
    
    ratios = background_votes / visible_counts
    
    # Determine which points to delete
    to_delete = ratios > threshold
    
    n_deleted = to_delete.sum().item()
    n_kept = n_points - n_deleted
    
    CONSOLE.print(f"[bold]Voting Results:[/bold]")
    CONSOLE.print(f"  Points to delete: {n_deleted:,} ({100*n_deleted/n_points:.1f}%)")
    CONSOLE.print(f"  Points to keep:   {n_kept:,} ({100*n_kept/n_points:.1f}%)")
    
    # Show distribution of voting ratios
    CONSOLE.print(f"\n[bold]Background Voting Ratio Distribution:[/bold]")
    for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        val = torch.quantile(ratios, q).item()
        CONSOLE.print(f"  {int(q*100):3d}% quantile: {val:.3f}")
    
    # Apply deletion mask to SuGaR model
    keep_mask = ~to_delete
    
    CONSOLE.print(f"\n[bold green]Applying culling to SuGaR model...[/bold green]")
    
    # Update all Gaussian parameters
    # Note: Different SuGaR versions may have different attribute names
    # We'll try to handle both _points and points patterns
    
    if hasattr(sugar_model, '_points'):
        sugar_model._points = sugar_model._points[keep_mask]
    
    if hasattr(sugar_model, '_scales'):
        sugar_model._scales = sugar_model._scales[keep_mask]
    
    if hasattr(sugar_model, '_quaternions'):
        sugar_model._quaternions = sugar_model._quaternions[keep_mask]
    
    if hasattr(sugar_model, 'all_densities'):
        sugar_model.all_densities = sugar_model.all_densities[keep_mask]
    
    # Handle SH coefficients
    if hasattr(sugar_model, '_sh_coordinates_dc'):
        sugar_model._sh_coordinates_dc = sugar_model._sh_coordinates_dc[keep_mask]
    
    if hasattr(sugar_model, '_sh_coordinates_rest'):
        sugar_model._sh_coordinates_rest = sugar_model._sh_coordinates_rest[keep_mask]
    
    # Handle optional attributes
    if hasattr(sugar_model, '_features_dc'):
        sugar_model._features_dc = sugar_model._features_dc[keep_mask]
    
    if hasattr(sugar_model, '_features_rest'):
        sugar_model._features_rest = sugar_model._features_rest[keep_mask]
    
    if hasattr(sugar_model, '_opacity'):
        sugar_model._opacity = sugar_model._opacity[keep_mask]
    
    # Update KNN tracking if present
    if hasattr(sugar_model, 'knn_dists') and sugar_model.knn_dists is not None:
        sugar_model.knn_dists = sugar_model.knn_dists[keep_mask]
    
    if hasattr(sugar_model, 'knn_idx') and sugar_model.knn_idx is not None:
        # KNN indices need remapping, safer to reset
        sugar_model.knn_idx = None
        sugar_model.knn_dists = None
    
    CONSOLE.print(f"[bold green]✓ Culling complete! Final Gaussian count: {n_kept:,}[/bold green]")
    CONSOLE.print("="*80 + "\n")
    
    # Save debug statistics
    if debug_output_dir:
        debug_dir = Path(debug_output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'initial_gaussians': n_points,
            'deleted_gaussians': n_deleted,
            'kept_gaussians': n_kept,
            'deletion_ratio': n_deleted / n_points,
            'threshold': threshold,
            'processed_views': processed_views,
        }
        
        import json
        with open(debug_dir / 'culling_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        CONSOLE.print(f"Debug statistics saved to {debug_dir / 'culling_stats.json'}")
