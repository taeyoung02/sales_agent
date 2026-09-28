"""
Debugging and Visualization Tool for SAM + CLIP Fine-tuning

이 스크립트는 다음을 시각화합니다:
1. SAM이 생성한 마스크들
2. Context-Aware Blur Cropping 적용 전/후 비교
3. CLIP이 각 마스크에 할당한 라벨과 신뢰도
4. Ground Truth와의 비교 (available시)
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple
import json
from pathlib import Path

from model import SAMCLIPModel
from dataset import VehicleSegDataset
from load_finetuned_sam import load_finetuned_sam


def load_model(checkpoint_path: str, sam_checkpoint: str, device: str = "cuda") -> SAMCLIPModel:
    """학습된 모델 로드"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load fine-tuned SAM separately
    print(f"Loading pretrained SAM from: {sam_checkpoint}")
    finetuned_sam = load_finetuned_sam(
        checkpoint_path=sam_checkpoint,
        sam_model_type="vit_b",
        device=device
    )
    
    model = SAMCLIPModel(
        sam_checkpoint=None,  # Don't load again, we'll set it manually
        sam_model_type="vit_b",
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
        num_classes=len(VehicleSegDataset.LABELS),
        freeze_sam=True,
        device=device
    )
    
    # Replace SAM with fine-tuned version
    model.sam = finetuned_sam
    model.sam_predictor.model = finetuned_sam
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model weights loaded")
    else:
        print("⚠ Checkpoint not found, using initialized model")
    
    model.eval()
    return model


def visualize_sam_masks(
    image: np.ndarray,
    masks: List[Dict],
    save_path: str = None,
    show: bool = True
):
    """
    SAM이 생성한 모든 마스크를 시각화
    
    Args:
        image: RGB image (H, W, 3)
        masks: List of SAM mask dictionaries
        save_path: 저장 경로 (optional)
        show: plt.show() 호출 여부
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'SAM Automatic Mask Generation ({len(masks)} masks)', fontsize=16, fontweight='bold')
    
    # 1. Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. All masks overlaid
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = plt.cm.tab20(i % 20)[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        overlay[mask['segmentation']] = overlay[mask['segmentation']] * 0.6 + color * 0.4
    
    axes[0, 1].imshow(overlay.astype(np.uint8))
    axes[0, 1].set_title(f'All Masks Overlay ({len(masks)} masks)', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Mask boundaries
    boundaries = np.zeros_like(image)
    for mask in masks:
        contours, _ = cv2.findContours(
            mask['segmentation'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(boundaries, contours, -1, (255, 255, 0), 2)
    
    result = cv2.addWeighted(image, 0.7, boundaries, 0.3, 0)
    axes[0, 2].imshow(result)
    axes[0, 2].set_title('Mask Boundaries', fontsize=14)
    axes[0, 2].axis('off')
    
    # 4. Stability scores heatmap
    if len(masks) > 0 and 'stability_score' in masks[0]:
        stability_map = np.zeros(image.shape[:2], dtype=np.float32)
        for mask in masks:
            stability_map[mask['segmentation']] = mask['stability_score']
        
        im = axes[1, 0].imshow(stability_map, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Stability Scores', fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # 5. Predicted IoU heatmap
    if len(masks) > 0 and 'predicted_iou' in masks[0]:
        iou_map = np.zeros(image.shape[:2], dtype=np.float32)
        for mask in masks:
            iou_map[mask['segmentation']] = mask['predicted_iou']
        
        im = axes[1, 1].imshow(iou_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('Predicted IoU', fontsize=14)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # 6. Mask size distribution
    if len(masks) > 0:
        mask_areas = [mask['area'] for mask in masks]
        axes[1, 2].hist(mask_areas, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Mask Size Distribution', fontsize=14)
        axes[1, 2].set_xlabel('Area (pixels)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ SAM masks visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_context_aware_cropping(
    image: np.ndarray,
    masks: List[Dict],
    max_samples: int = 6,
    save_path: str = None,
    show: bool = True
):
    """
    Context-Aware Blur Cropping 전/후 비교 시각화
    
    Args:
        image: RGB image
        masks: SAM masks
        max_samples: 시각화할 최대 마스크 수
        save_path: 저장 경로
        show: plt.show() 호출 여부
    """
    from model import SAMCLIPModel
    
    num_samples = min(len(masks), max_samples)
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Context-Aware Blur Cropping Comparison', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        mask_dict = masks[idx]
        mask = mask_dict['segmentation']
        
        # Column 1: Original image with bbox
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(image)
        
        # Draw original bbox
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='red', facecolor='none', label='Original BBox'
            )
            ax1.add_patch(rect)
            
            # Draw expanded bbox (30%)
            w, h = x_max - x_min, y_max - y_min
            pad_w, pad_h = int(w * 0.3), int(h * 0.3)
            x1 = max(0, x_min - pad_w)
            y1 = max(0, y_min - pad_h)
            x2 = min(image.shape[1], x_max + pad_w)
            y2 = min(image.shape[0], y_max + pad_h)
            
            rect_expanded = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='yellow', facecolor='none', 
                linestyle='--', label='Expanded BBox (+30%)'
            )
            ax1.add_patch(rect_expanded)
        
        ax1.set_title(f'Mask #{idx+1}\nArea: {mask_dict.get("area", 0):.0f} px', fontsize=12)
        ax1.axis('off')
        ax1.legend(loc='upper right', fontsize=8)
        
        # Column 2: Mask only
        ax2 = fig.add_subplot(gs[idx, 1])
        mask_vis = np.zeros_like(image)
        mask_vis[mask] = image[mask]
        ax2.imshow(mask_vis)
        ax2.set_title('Mask Region', fontsize=12)
        ax2.axis('off')
        
        # Column 3: Tight crop (original method)
        ax3 = fig.add_subplot(gs[idx, 2])
        if len(y_indices) > 0:
            tight_crop = image[y_min:y_max+1, x_min:x_max+1].copy()
            # Mask out background
            tight_mask = mask[y_min:y_max+1, x_min:x_max+1]
            tight_crop[tight_mask == 0] = 0
            
            # Resize to 224x224 for visualization
            tight_crop_resized = cv2.resize(tight_crop, (224, 224))
            ax3.imshow(tight_crop_resized)
            ax3.set_title('Original Method\n(Tight Crop + Black BG)', fontsize=12)
        ax3.axis('off')
        
        # Column 4: Context-aware crop (NO blur - just expansion)
        ax4 = fig.add_subplot(gs[idx, 2])
        context_crop_no_blur = SAMCLIPModel._get_context_aware_crop(
            mask, image, expand_ratio=0.3, blur_strength=(1, 1)  # No blur
        )
        context_crop_no_blur_resized = cv2.resize(context_crop_no_blur, (224, 224))
        ax4.imshow(context_crop_no_blur_resized)
        ax4.set_title('Expanded Crop\n(+30% context, No Blur)', fontsize=12)
        ax4.axis('off')
        
        # Column 5: Full context-aware crop (WITH blur)
        ax5 = fig.add_subplot(gs[idx, 3])
        context_crop = SAMCLIPModel._get_context_aware_crop(
            mask, image, expand_ratio=0.3, blur_strength=(51, 51)
        )
        context_crop_resized = cv2.resize(context_crop, (224, 224))
        ax5.imshow(context_crop_resized)
        ax5.set_title('Context-Aware Blur\n(+30% + Blur BG)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Column 6: Difference heatmap
        ax6 = fig.add_subplot(gs[idx, 4])
        diff = np.abs(context_crop_resized.astype(float) - tight_crop_resized.astype(float)).mean(axis=2)
        im = ax6.imshow(diff, cmap='hot')
        ax6.set_title('Difference Heatmap', fontsize=12)
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Context-aware cropping comparison saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_clip_predictions(
    image: np.ndarray,
    masks: List[Dict],
    predictions: List[str],
    confidences: List[float],
    ground_truth: Dict = None,
    save_path: str = None,
    show: bool = True,
    top_k: int = 10
):
    """
    CLIP 분류 결과를 상세하게 시각화 (top-k predictions with scores)
    
    Args:
        image: RGB image
        masks: SAM masks
        predictions: CLIP predicted labels
        confidences: Prediction confidences
        ground_truth: GT annotations (optional)
        save_path: 저장 경로
        show: plt.show() 호출 여부
        top_k: 보여줄 top-k 마스크 수
    """
    from collections import Counter
    
    # Sort masks by confidence
    sorted_indices = np.argsort(confidences)[::-1][:top_k]
    num_show = min(len(sorted_indices), top_k, len(masks))
    
    # Calculate grid layout
    cols = 3
    rows = (num_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    fig.suptitle(f'Top-{num_show} CLIP Predictions (sorted by confidence)', fontsize=16, fontweight='bold')
    
    # Visualize each top-k mask
    for i, mask_idx in enumerate(sorted_indices[:num_show]):
        if i >= len(axes):
            break
            
        mask_dict = masks[mask_idx]
        pred_label = predictions[mask_idx]
        conf = confidences[mask_idx]
        
        # Create overlay
        overlay = image.copy()
        seg = mask_dict['segmentation']
        
        # Green overlay for mask
        overlay[seg] = overlay[seg] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        
        # Draw bounding box
        if 'bbox' in mask_dict:
            x, y, w, h = mask_dict['bbox']
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 3)
        
        axes[i].imshow(overlay)
        
        # Title with detailed info
        title = f"#{i+1} (Mask {mask_idx}): {pred_label}\n"
        title += f"Confidence: {conf:.4f}"
        
        # Check if this is "Others" (low confidence)
        if pred_label == "Others" or conf < 0.25:
            title += " [LOW CONF - Others]"
        
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ CLIP predictions visualization saved to: {save_path}")
    
    # Print detailed statistics
    print(f"\n  Detailed CLIP Prediction Statistics:")
    print(f"  {'='*60}")
    print(f"  Total masks: {len(masks)}")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    
    # Count predictions above/below threshold
    threshold = 0.25
    above_threshold = sum(1 for c in confidences if c >= threshold)
    below_threshold = len(confidences) - above_threshold
    print(f"  Above threshold ({threshold}): {above_threshold}")
    print(f"  Below threshold ({threshold}): {below_threshold}")
    
    # Label distribution
    label_counts = Counter(predictions)
    print(f"\n  Label Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(predictions) * 100
        avg_conf = np.mean([confidences[i] for i, p in enumerate(predictions) if p == label])
        print(f"    • {label:20s}: {count:3d} ({percentage:5.1f}%) - avg conf: {avg_conf:.4f}")
    
    print(f"  {'='*60}\n")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_ground_truth_comparison(
    image: np.ndarray,
    sam_masks: List[Dict],
    predictions: List[str],
    gt_annotation: Dict,
    dataset: VehicleSegDataset,
    save_path: str = None,
    show: bool = True
):
    """
    Ground Truth와 예측 결과 비교
    
    Args:
        image: RGB image
        sam_masks: SAM generated masks
        predictions: CLIP predictions
        gt_annotation: Ground truth annotation
        dataset: VehicleSegDataset instance
        save_path: 저장 경로
        show: plt.show() 호출 여부
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Ground Truth vs Predictions Comparison', fontsize=16, fontweight='bold')
    
    # Parse GT annotations
    h, w = image.shape[:2]
    gt_masks = []
    gt_labels = []
    
    for shape in gt_annotation.get('shapes', []):
        label = shape.get('label', 'Others')
        points = shape.get('points', [])
        
        if len(points) >= 3:
            mask = dataset._polygon_to_mask(points, h, w)
            gt_masks.append(mask)
            gt_labels.append(label)
    
    # 1. Ground Truth
    gt_overlay = image.copy()
    for i, (mask, label) in enumerate(zip(gt_masks, gt_labels)):
        color = plt.cm.tab20(i % 20)[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        gt_overlay[mask > 0] = gt_overlay[mask > 0] * 0.5 + color * 0.5
    
    axes[0, 0].imshow(gt_overlay.astype(np.uint8))
    axes[0, 0].set_title(f'Ground Truth ({len(gt_masks)} masks)', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. SAM Predictions
    pred_overlay = image.copy()
    for i, (mask_dict, pred) in enumerate(zip(sam_masks, predictions)):
        mask = mask_dict['segmentation']
        color = plt.cm.tab20(i % 20)[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        pred_overlay[mask] = pred_overlay[mask] * 0.5 + color * 0.5
    
    axes[0, 1].imshow(pred_overlay.astype(np.uint8))
    axes[0, 1].set_title(f'SAM + CLIP Predictions ({len(sam_masks)} masks)', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Matched masks (IoU-based)
    matched_vis = image.copy()
    matched_count = 0
    
    for sam_mask_dict, pred_label in zip(sam_masks, predictions):
        sam_mask = sam_mask_dict['segmentation']
        
        # Find best matching GT mask
        max_iou = 0.0
        best_gt_label = None
        
        for gt_mask, gt_label in zip(gt_masks, gt_labels):
            iou = dataset.compute_iou(sam_mask, gt_mask)
            if iou > max_iou:
                max_iou = iou
                best_gt_label = gt_label
        
        # Color code: green if match, red if mismatch
        if max_iou >= 0.5:
            matched_count += 1
            if pred_label == best_gt_label:
                color = np.array([0, 255, 0])  # Green - correct
            else:
                color = np.array([255, 0, 0])  # Red - wrong label
        else:
            color = np.array([128, 128, 128])  # Gray - no GT match
        
        matched_vis[sam_mask] = matched_vis[sam_mask] * 0.6 + color * 0.4
    
    axes[1, 0].imshow(matched_vis.astype(np.uint8))
    axes[1, 0].set_title(
        f'Matching Visualization\nGreen=Correct, Red=Wrong Label, Gray=No GT Match',
        fontsize=14
    )
    axes[1, 0].axis('off')
    
    # 4. Confusion matrix
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    
    for sam_mask_dict, pred_label in zip(sam_masks, predictions):
        sam_mask = sam_mask_dict['segmentation']
        
        max_iou = 0.0
        best_gt_label = "No GT"
        
        for gt_mask, gt_label in zip(gt_masks, gt_labels):
            iou = dataset.compute_iou(sam_mask, gt_mask)
            if iou > max_iou:
                max_iou = iou
                best_gt_label = gt_label
        
        if max_iou >= 0.5:
            confusion[best_gt_label][pred_label] += 1
    
    # Plot confusion matrix
    if confusion:
        all_labels = sorted(set(list(confusion.keys()) + 
                               [pred for preds in confusion.values() for pred in preds.keys()]))
        
        conf_matrix = np.zeros((len(all_labels), len(all_labels)))
        for i, gt_label in enumerate(all_labels):
            for j, pred_label in enumerate(all_labels):
                conf_matrix[i, j] = confusion[gt_label][pred_label]
        
        im = axes[1, 1].imshow(conf_matrix, cmap='Blues', aspect='auto')
        axes[1, 1].set_xticks(range(len(all_labels)))
        axes[1, 1].set_yticks(range(len(all_labels)))
        axes[1, 1].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_yticklabels(all_labels, fontsize=9)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=12)
        axes[1, 1].set_ylabel('Ground Truth Label', fontsize=12)
        axes[1, 1].set_title('Confusion Matrix (IoU≥0.5)', fontsize=14)
        
        # Add text annotations
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                count = int(conf_matrix[i, j])
                if count > 0:
                    axes[1, 1].text(j, i, str(count), ha='center', va='center',
                                   color='white' if count > conf_matrix.max()/2 else 'black',
                                   fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ GT comparison visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def debug_single_image(
    model: SAMCLIPModel,
    image_path: str,
    dataset: VehicleSegDataset = None,
    output_dir: str = "debug_output",
    show: bool = False
):
    """
    단일 이미지에 대한 전체 디버깅 파이프라인 실행
    
    Args:
        model: Trained SAMCLIPModel
        image_path: 이미지 경로
        dataset: VehicleSegDataset (GT 비교용, optional)
        output_dir: 출력 디렉토리
        show: matplotlib show 호출 여부
    """
    print(f"\n{'='*60}")
    print(f"Debugging Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: SAM mask generation
    print("Step 1: Generating SAM masks...")
    sam_masks = model.generate_sam_masks(image)
    print(f"  ✓ Generated {len(sam_masks)} masks")
    
    visualize_sam_masks(
        image, sam_masks,
        save_path=f"{output_dir}/{image_name}_1_sam_masks.png",
        show=show
    )
    
    # Step 2: Context-aware cropping comparison
    print("\nStep 2: Visualizing context-aware cropping...")
    visualize_context_aware_cropping(
        image, sam_masks, max_samples=6,
        save_path=f"{output_dir}/{image_name}_2_context_aware_cropping.png",
        show=show
    )
    
    # Step 3: CLIP predictions
    print("\nStep 3: Running CLIP classification...")
    predictions, confidences, _ = model.predict(image, return_masks=False)
    print(f"  ✓ Classified {len(predictions)} masks")
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    
    visualize_clip_predictions(
        image, sam_masks, predictions, confidences,
        save_path=f"{output_dir}/{image_name}_3_clip_predictions.png",
        show=show
    )
    
    # Step 4: Ground truth comparison (if available)
    if dataset is not None:
        print("\nStep 4: Comparing with ground truth...")
        
        # Find annotation file
        annotation_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 
                                     'annotations', os.path.basename(os.path.dirname(image_path)))
        annotation_path = os.path.join(annotation_dir, f"{image_name}.json")
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                gt_annotation = json.load(f)
            
            visualize_ground_truth_comparison(
                image, sam_masks, predictions, gt_annotation, dataset,
                save_path=f"{output_dir}/{image_name}_4_gt_comparison.png",
                show=show
            )
            print("  ✓ Ground truth comparison complete")
        else:
            print(f"  ⚠ Ground truth not found: {annotation_path}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  - SAM masks: {len(sam_masks)}")
    print(f"  - Average confidence: {np.mean(confidences):.3f}")
    print(f"  - Label distribution:")
    
    from collections import Counter
    label_counts = Counter(predictions)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    • {label}: {count}")
    
    print(f"\n  ✓ All visualizations saved to: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='SAM + CLIP Debugging & Visualization')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='outputs/sam_clip_20251230_095519/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Path to VehicleSeg10K dataset (for GT comparison)')
    parser.add_argument('--output-dir', type=str, default='debug_output',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (default: save only)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.sam_checkpoint, args.device)
    
    # Load dataset (optional, for GT comparison)
    dataset = None
    if args.data_root:
        dataset = VehicleSegDataset(
            root_dir=args.data_root,
            split='valid',  # Use validation split for debugging
            transform=None
        )
        print(f"✓ Loaded dataset from: {args.data_root}")
    
    # Run debugging
    debug_single_image(
        model=model,
        image_path=args.image,
        dataset=dataset,
        output_dir=args.output_dir,
        show=args.show
    )


if __name__ == '__main__':
    main()
