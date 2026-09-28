"""
SAM Fine-tuning Training Script
Train mask decoder and prompt encoder on VehicleSeg10K dataset
"""
import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sam_dataset import SAMDataset, sam_collate_fn
from sam_model import SAMFineTuner
from sam_loss import SAMSegmentationLoss


def train_one_epoch(
    model: SAMFineTuner,
    dataloader: DataLoader,
    criterion: SAMSegmentationLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    max_samples_per_epoch: int = None,
    max_masks_per_image: int = 10,
    scaler=None
) -> dict:
    """Train for one epoch with speed optimizations"""
    model.train()
    
    # Only mask decoder and prompt encoder are in train mode
    model.sam.image_encoder.eval()  # Keep frozen
    
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    total_iou_loss = 0.0
    total_iou_score = 0.0
    num_masks = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Check if we've reached max samples
        if max_samples_per_epoch and num_masks >= max_samples_per_epoch:
            break
        
        batch_loss = 0.0
        batch_masks = 0
        
        # Process each sample in batch (batch size = 1 for SAM)
        for sample in batch:
            image = sample['image']  # (H, W, 3) numpy
            gt_masks = sample['masks']  # List of (H, W) binary masks
            prompts_list = sample['prompts']  # List of prompt lists
            
            # [CRITICAL OPTIMIZATION] Pre-encode image ONCE per image (not per mask!)
            # This is the BIGGEST bottleneck - image encoding takes ~80% of time
            orig_h, orig_w = image.shape[:2]
            image_tensor = model.preprocess_image(image)
            with torch.no_grad():
                image_embeddings = model.sam.image_encoder(image_tensor)
            
            # [OPTIMIZATION] Limit number of masks per image for speed
            num_masks_to_process = min(len(gt_masks), max_masks_per_image)
            
            # Process each mask with its prompts
            for idx in range(num_masks_to_process):
                gt_mask = gt_masks[idx]
                prompts = prompts_list[idx]
                # Use first (only) prompt
                prompt = prompts[0]
                
                bbox = prompt['bbox']
                points = prompt['points']
                point_labels = prompt['point_labels']
                
                # [OPTIMIZATION] Mixed Precision Training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass with pre-computed image embeddings
                        pred_masks, pred_iou = model.forward_with_embeddings(
                            image_embeddings=image_embeddings,
                            orig_size=(orig_h, orig_w),
                            bbox=bbox,
                            points=points,
                            point_labels=point_labels,
                            return_logits=True
                        )
                        
                        # Prepare ground truth
                        gt_tensor = model.preprocess_mask(gt_mask, (orig_h, orig_w))
                        
                        # Calculate loss
                        losses = criterion(pred_masks, gt_tensor, pred_iou)
                        loss = losses['total']
                    
                    # Backward with scaler
                    scaler.scale(loss).backward()
                else:
                    # Forward pass with pre-computed image embeddings
                    pred_masks, pred_iou = model.forward_with_embeddings(
                        image_embeddings=image_embeddings,
                        orig_size=(orig_h, orig_w),
                        bbox=bbox,
                        points=points,
                        point_labels=point_labels,
                        return_logits=True
                    )
                    
                    # Prepare ground truth
                    gt_tensor = model.preprocess_mask(gt_mask, (orig_h, orig_w))
                    
                    # Calculate loss
                    losses = criterion(pred_masks, gt_tensor, pred_iou)
                    loss = losses['total']
                    
                    # Backward
                    loss.backward()
                
                # Accumulate metrics
                batch_loss += loss.item()
                total_focal += losses['focal']
                total_dice += losses['dice']
                total_iou_loss += losses['iou']
                
                # Calculate actual IoU score
                with torch.no_grad():
                    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                    intersection = (pred_binary * gt_tensor).sum()
                    union = pred_binary.sum() + gt_tensor.sum() - intersection
                    iou_score = (intersection / (union + 1e-6)).item()
                    total_iou_score += iou_score
                
                batch_masks += 1
                num_masks += 1
        
        # Optimizer step
        if batch_masks > 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            total_loss += batch_loss
            
            # Update progress bar
            avg_loss = batch_loss / batch_masks
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'masks': num_masks
            })
    
    # Epoch metrics
    if num_masks > 0:
        avg_loss = total_loss / num_masks
        avg_focal = total_focal / num_masks
        avg_dice = total_dice / num_masks
        avg_iou_loss = total_iou_loss / num_masks
        avg_iou_score = total_iou_score / num_masks
    else:
        avg_loss = avg_focal = avg_dice = avg_iou_loss = avg_iou_score = 0.0
    
    # TensorBoard logging
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Focal', avg_focal, epoch)
    writer.add_scalar('Train/Dice', avg_dice, epoch)
    writer.add_scalar('Train/IoU_Loss', avg_iou_loss, epoch)
    writer.add_scalar('Train/IoU_Score', avg_iou_score, epoch)
    
    return {
        'loss': avg_loss,
        'focal': avg_focal,
        'dice': avg_dice,
        'iou_loss': avg_iou_loss,
        'iou_score': avg_iou_score,
        'num_masks': num_masks
    }


@torch.no_grad()
def validate(
    model: SAMFineTuner,
    dataloader: DataLoader,
    criterion: SAMSegmentationLoss,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    max_samples: int = None,
    max_masks_per_image: int = 10
) -> dict:
    """Validation with mask limit"""
    model.eval()
    
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    total_iou_loss = 0.0
    total_iou_score = 0.0
    num_masks = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")
    
    for batch_idx, batch in enumerate(pbar):
        # Limit validation samples
        if max_samples and num_masks >= max_samples:
            break
        
        for sample in batch:
            image = sample['image']
            gt_masks = sample['masks']
            prompts_list = sample['prompts']
            
            # [CRITICAL OPTIMIZATION] Pre-encode image ONCE per image
            orig_h, orig_w = image.shape[:2]
            image_tensor = model.preprocess_image(image)
            image_embeddings = model.sam.image_encoder(image_tensor)
            
            # [OPTIMIZATION] Limit masks
            num_masks_to_process = min(len(gt_masks), max_masks_per_image)
            
            for idx in range(num_masks_to_process):
                gt_mask = gt_masks[idx]
                prompts = prompts_list[idx]
                # Use first prompt for validation (consistent)
                prompt = prompts[0]
                
                bbox = prompt['bbox']
                points = prompt['points']
                point_labels = prompt['point_labels']
                
                # Forward pass with pre-computed embeddings
                pred_masks, pred_iou = model.forward_with_embeddings(
                    image_embeddings=image_embeddings,
                    orig_size=(orig_h, orig_w),
                    bbox=bbox,
                    points=points,
                    point_labels=point_labels,
                    return_logits=True
                )
                
                # Ground truth
                gt_tensor = model.preprocess_mask(gt_mask, (orig_h, orig_w))
                
                # Loss
                losses = criterion(pred_masks, gt_tensor, pred_iou)
                
                total_loss += losses['total'].item()
                total_focal += losses['focal']
                total_dice += losses['dice']
                total_iou_loss += losses['iou']
                
                # IoU score
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                intersection = (pred_binary * gt_tensor).sum()
                union = pred_binary.sum() + gt_tensor.sum() - intersection
                iou_score = (intersection / (union + 1e-6)).item()
                total_iou_score += iou_score
                
                num_masks += 1
        
        # Update progress
        if num_masks > 0:
            pbar.set_postfix({
                'loss': f'{total_loss/num_masks:.4f}',
                'iou': f'{total_iou_score/num_masks:.4f}'
            })
    
    # Metrics
    if num_masks > 0:
        avg_loss = total_loss / num_masks
        avg_focal = total_focal / num_masks
        avg_dice = total_dice / num_masks
        avg_iou_loss = total_iou_loss / num_masks
        avg_iou_score = total_iou_score / num_masks
    else:
        avg_loss = avg_focal = avg_dice = avg_iou_loss = avg_iou_score = 0.0
    
    # TensorBoard
    writer.add_scalar('Valid/Loss', avg_loss, epoch)
    writer.add_scalar('Valid/Focal', avg_focal, epoch)
    writer.add_scalar('Valid/Dice', avg_dice, epoch)
    writer.add_scalar('Valid/IoU_Loss', avg_iou_loss, epoch)
    writer.add_scalar('Valid/IoU_Score', avg_iou_score, epoch)
    
    return {
        'loss': avg_loss,
        'focal': avg_focal,
        'dice': avg_dice,
        'iou_loss': avg_iou_loss,
        'iou_score': avg_iou_score,
        'num_masks': num_masks
    }


def main(args):
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sam_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "logs")
    writer = SummaryWriter(log_dir)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"TensorBoard logs: {log_dir}")
    
    # Dataset
    print("\nLoading datasets...")
    train_dataset = SAMDataset(
        root_dir=args.dataset_root,
        split='train',
        num_prompts_per_mask=args.num_prompts,
        use_bbox_prompt=args.use_bbox,
        use_point_prompt=args.use_point,
        augment=True
    )
    
    valid_dataset = SAMDataset(
        root_dir=args.dataset_root,
        split='valid',
        num_prompts_per_mask=args.num_prompts,
        use_bbox_prompt=args.use_bbox,
        use_point_prompt=args.use_point,
        augment=False
    )
    
    # DataLoader (batch_size=1 for SAM)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sam_collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=sam_collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Model
    print("\nBuilding SAM fine-tuning model...")
    model = SAMFineTuner(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        freeze_image_encoder=True,
        device=device
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,} ({count/1e6:.2f}M)")
    
    # Loss
    criterion = SAMSegmentationLoss(
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        iou_weight=args.iou_weight
    )
    
    print(f"\n✓ Loss configuration:")
    print(f"  - Focal weight: {args.focal_weight}")
    print(f"  - Dice weight: {args.dice_weight}")
    print(f"  - IoU weight: {args.iou_weight}")
    
    # Optimizer (only trainable parameters)
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Mixed Precision Training (AMP) - Always enabled for CUDA
    scaler = None
    if device.startswith('cuda'):
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("\n✓ Mixed Precision Training (AMP) enabled")
        print(f"  GPU Memory optimization: FP16 computations")
    else:
        print("\n✗ Mixed Precision disabled (CPU mode)")
    
    # Speed optimization settings
    print(f"\n✓ Speed Optimizations:")
    print(f"  - Max masks per image: {args.max_masks_per_image}")
    print(f"  - Prompts per mask: 1 (random from {args.num_prompts})")
    print(f"  - Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
    
    # Training loop
    print("\nStarting SAM fine-tuning...")
    best_val_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            max_samples_per_epoch=args.max_train_samples,
            max_masks_per_image=args.max_masks_per_image,
            scaler=scaler
        )
        
        print(f"\n[Train] Epoch {epoch}:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Focal: {train_metrics['focal']:.4f}")
        print(f"  Dice: {train_metrics['dice']:.4f}")
        print(f"  IoU Score: {train_metrics['iou_score']:.4f}")
        print(f"  Processed: {train_metrics['num_masks']} masks")
        
        # Validate
        val_metrics = validate(
            model, valid_loader, criterion, device, epoch, writer,
            max_samples=args.max_valid_samples,
            max_masks_per_image=args.max_masks_per_image
        )
        
        print(f"\n[Valid] Epoch {epoch}:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Focal: {val_metrics['focal']:.4f}")
        print(f"  Dice: {val_metrics['dice']:.4f}")
        print(f"  IoU Score: {val_metrics['iou_score']:.4f}")
        print(f"  Processed: {val_metrics['num_masks']} masks")
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        print(f"\n  Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['iou_score'] > best_val_iou:
            best_val_iou = val_metrics['iou_score']
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_loss': val_metrics['loss'],
            }, best_model_path)
            print(f"  ✓ Best model saved (IoU: {best_val_iou:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 1 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['iou_score'],
            }, checkpoint_path)
    
    # Final save
    final_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.sam.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Best validation IoU: {best_val_iou:.4f}")
    print(f"  Models saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM Fine-tuning on VehicleSeg10K")
    
    # Dataset
    parser.add_argument('--dataset_root', type=str, default='VehicleSeg10K',
                       help='Path to VehicleSeg10K dataset')
    
    # Model
    parser.add_argument('--sam_checkpoint', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--focal_weight', type=float, default=1.0,
                       help='Focal loss weight')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Dice loss weight')
    parser.add_argument('--iou_weight', type=float, default=0.5,
                       help='IoU loss weight')
    
    # Prompts
    parser.add_argument('--num_prompts', type=int, default=5,
                       help='Number of prompts per mask')
    parser.add_argument('--use_bbox', action='store_true', default=True,
                       help='Use bounding box prompts')
    parser.add_argument('--use_point', action='store_true', default=True,
                       help='Use point prompts')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    
    # Performance
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Max training samples per epoch (for quick testing)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Max validation samples (for quick testing)')
    parser.add_argument('--max_masks_per_image', type=int, default=10,
                       help='Max masks to process per image (speed optimization)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    
    args = parser.parse_args()
    
    main(args)
