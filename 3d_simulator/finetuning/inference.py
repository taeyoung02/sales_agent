"""
Inference script for trained SAM + CLIP model
"""
import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from model import SAMCLIPModel
from dataset import VehicleSegDataset


def visualize_predictions(
    image: np.ndarray,
    masks: list,
    labels: list,
    confidences: list,
    save_path: str = None,
    show: bool = True
):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(image)
    
    # Color map for labels
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(VehicleSegDataset.LABELS)))
    label_colors = {label: colors[i] for i, label in enumerate(VehicleSegDataset.LABELS)}
    
    # Draw masks
    for mask, label, conf in zip(masks, labels, confidences):
        # Create colored mask
        color = label_colors.get(label, (1, 0, 0))
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = (*color[:3], 0.5)
        
        axes[1].imshow(colored_mask)
    
    axes[1].set_title('Predictions')
    axes[1].axis('off')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=label_colors[label][:3], label=f'{label} ({sum(1 for l in labels if l == label)})')
        for label in set(labels)
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference with SAM + CLIP model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--output-dir', type=str, default='./outputs/inference',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        torch.cuda.set_device(args.gpu_id)
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = SAMCLIPModel(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type='vit_b',
        clip_model_name='ViT-B-32',
        clip_pretrained='openai',
        num_classes=len(VehicleSegDataset.LABELS),
        freeze_sam=True,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inference
    print("Running inference...")
    labels, confidences, masks = model.predict(image, return_masks=True)
    
    print(f"\nDetected {len(labels)} masks:")
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        print(f"  {i+1}. {label}: {conf:.3f}")
    
    # Visualize
    image_name = os.path.basename(args.image).rsplit('.', 1)[0]
    save_path = os.path.join(args.output_dir, f'{image_name}_result.png')
    
    visualize_predictions(
        image,
        masks,
        labels,
        confidences,
        save_path=save_path,
        show=args.show
    )
    
    # Save results as JSON
    import json
    results = {
        'image_path': args.image,
        'num_masks': len(labels),
        'predictions': [
            {
                'label': label,
                'confidence': float(conf),
                'mask_area': int(mask.sum())
            }
            for label, conf, mask in zip(labels, confidences, masks)
        ]
    }
    
    json_path = os.path.join(args.output_dir, f'{image_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")


if __name__ == '__main__':
    main()
