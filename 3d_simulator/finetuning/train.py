"""
Training script for SAM + CLIP model on VehicleSeg10K dataset
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

from dataset import VehicleSegDataset, collate_fn
from model import SAMCLIPModel
from loss import BackgroundAwareLoss

# Performance optimization
try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


def train_one_epoch(
    model: SAMCLIPModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    scaler=None,
    accumulation_steps: int = 1
) -> dict:
    """한 에폭 학습 (Mixed Precision + Gradient Accumulation)"""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, samples in enumerate(pbar):
        batch_loss = 0.0
        batch_correct = 0
        batch_samples = 0
        
        for sample in samples:
            image_array = sample['image_array']  # (H, W, 3) numpy
            gt_masks = sample['masks']  # List of (H, W) numpy masks - includes background "Others"
            gt_labels = sample['labels']  # List of label names
            gt_label_indices = sample['label_indices']  # List of label indices
            
            # SAM으로 마스크 생성
            sam_masks = model.generate_sam_masks(image_array)
            
            if len(sam_masks) == 0:
                continue
            
            # SAM 마스크를 GT와 매칭하여 라벨 할당
            matched_labels = []
            matched_indices = []
            valid_sam_masks = []
            
            for sam_mask_dict in sam_masks:
                sam_mask = sam_mask_dict['segmentation']
                
                # GT와 매칭 (background "Others" 포함)
                max_iou = 0.0
                matched_label = "Others"  # default: background
                matched_idx = VehicleSegDataset.LABELS.index("Others")
                
                for gt_mask, gt_label, gt_idx in zip(gt_masks, gt_labels, gt_label_indices):
                    iou = VehicleSegDataset.compute_iou(sam_mask, gt_mask)
                    if iou > max_iou:
                        max_iou = iou
                        matched_label = gt_label
                        matched_idx = gt_idx
                
                # IoU threshold (0.3) 이상인 것만 사용
                # 낮은 threshold: SAM 마스크도 배경과 매칭될 수 있도록
                if max_iou >= 0.3:
                    matched_labels.append(matched_label)
                    matched_indices.append(matched_idx)
                    valid_sam_masks.append(sam_mask)
            
            if len(valid_sam_masks) == 0:
                continue
            
            # 이미지를 tensor로 변환
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(device)
            
            # 마스크를 tensor로 변환
            mask_tensors = [
                torch.from_numpy(m.astype(np.float32)).to(device)
                for m in valid_sam_masks
            ]
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast(device_type='cuda'):
                    logits, features = model(image_tensor, mask_tensors, return_features=True)
                    labels = torch.tensor(matched_indices, dtype=torch.long, device=device)
                    
                    if isinstance(criterion, BackgroundAwareLoss):
                        loss = criterion(logits, labels, features, model.background_embedding)
                    else:
                        loss = criterion(logits, labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward with gradient scaling
                scaler.scale(loss).backward()
            else:
                # Standard precision
                logits, features = model(image_tensor, mask_tensors, return_features=True)
                labels = torch.tensor(matched_indices, dtype=torch.long, device=device)
                
                if isinstance(criterion, BackgroundAwareLoss):
                    loss = criterion(logits, labels, features, model.background_embedding)
                else:
                    loss = criterion(logits, labels)
                
                loss = loss / accumulation_steps
                loss.backward()
            
            # Metrics
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum().item()
            
            batch_loss += loss.item() * accumulation_steps  # Unscale for logging
            batch_correct += correct
            batch_samples += len(labels)
        
        # Optimizer step after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        if batch_samples > 0:
            total_loss += batch_loss
            total_correct += batch_correct
            total_samples += batch_samples
            
            # Progress bar 업데이트
            accuracy = 100. * batch_correct / batch_samples
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })
    
    # Epoch metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    
    # TensorBoard logging
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', avg_acc, epoch)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'total_samples': total_samples
    }


def validate(
    model: SAMCLIPModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    writer: SummaryWriter
) -> dict:
    """Validation"""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")
    
    with torch.no_grad():
        for batch_idx, samples in enumerate(pbar):
            batch_loss = 0.0
            batch_correct = 0
            batch_samples = 0
            
            for sample in samples:
                image_array = sample['image_array']
                gt_masks = sample['masks']  # includes background "Others"
                gt_labels = sample['labels']
                gt_label_indices = sample['label_indices']
                
                # SAM으로 마스크 생성
                sam_masks = model.generate_sam_masks(image_array)
                
                if len(sam_masks) == 0:
                    continue
                
                # SAM 마스크를 GT와 매칭
                matched_indices = []
                valid_sam_masks = []
                
                for sam_mask_dict in sam_masks:
                    sam_mask = sam_mask_dict['segmentation']
                    
                    max_iou = 0.0
                    matched_idx = VehicleSegDataset.LABELS.index("Others")
                    
                    for gt_mask, gt_label, gt_idx in zip(gt_masks, gt_labels, gt_label_indices):
                        iou = VehicleSegDataset.compute_iou(sam_mask, gt_mask)
                        if iou > max_iou:
                            max_iou = iou
                            matched_idx = gt_idx
                    
                    if max_iou >= 0.3:
                        matched_indices.append(matched_idx)
                        valid_sam_masks.append(sam_mask)
                
                if len(valid_sam_masks) == 0:
                    continue
                
                # Tensor 변환
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.to(device)
                
                mask_tensors = [
                    torch.from_numpy(m.astype(np.float32)).to(device)
                    for m in valid_sam_masks
                ]
                
                # Forward pass
                with torch.no_grad():
                    logits, features = model(image_tensor, mask_tensors, return_features=True)
                
                labels = torch.tensor(matched_indices, dtype=torch.long, device=device)
                
                # Loss 계산
                if isinstance(criterion, BackgroundAwareLoss):
                    loss = criterion(logits, labels, features, model.background_embedding)
                else:
                    loss = criterion(logits, labels)
                
                # Metrics
                _, predicted = logits.max(1)
                correct = predicted.eq(labels).sum().item()
                
                batch_loss += loss.item()
                batch_correct += correct
                batch_samples += len(labels)
            
            if batch_samples > 0:
                total_loss += batch_loss
                total_correct += batch_correct
                total_samples += batch_samples
                
                accuracy = 100. * batch_correct / batch_samples
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    
    # TensorBoard logging
    writer.add_scalar('Valid/Loss', avg_loss, epoch)
    writer.add_scalar('Valid/Accuracy', avg_acc, epoch)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'total_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Train SAM + CLIP on VehicleSeg10K')
    
    # Dataset args
    parser.add_argument('--data-root', type=str, default='./VehicleSeg10K',
                        help='Root directory of VehicleSeg10K dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model args
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--sam-model-type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')
    parser.add_argument('--use-finetuned-sam', action='store_true', default=False,
                        help='Use fine-tuned SAM model instead of pretrained')
    parser.add_argument('--finetuned-sam-dir', type=str, default=None,
                        help='Path to fine-tuned SAM output directory (e.g., outputs/sam_20260101_003032)')
    parser.add_argument('--clip-model', type=str, default='ViT-B-32',
                        help='CLIP model name')
    parser.add_argument('--clip-pretrained', type=str, default='openai',
                        help='CLIP pretrained weights')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--save-dir', type=str, default='./outputs',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save-every', type=int, default=5,
                        help='(Deprecated) Checkpoints are now saved every epoch automatically')
    
    # GPU args
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    
    # Performance optimization args
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) for faster training')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='Disable AMP')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--compile-model', action='store_true', default=False,
                        help='Compile model with torch.compile() (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'sam_clip_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # Dataset
    print("Loading datasets...")
    train_dataset = VehicleSegDataset(
        root_dir=args.data_root,
        split='train',
        iou_threshold=0.5
    )
    
    valid_dataset = VehicleSegDataset(
        root_dir=args.data_root,
        split='valid',
        iou_threshold=0.5
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    
    # DataLoader with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.startswith('cuda') else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
        pin_memory=True if device.startswith('cuda') else False
    )
    
    # Model
    print("Building model...")
    model = SAMCLIPModel(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        num_classes=len(VehicleSegDataset.LABELS),
        freeze_sam=True,
        use_finetuned_sam=args.use_finetuned_sam,
        finetuned_sam_dir=args.finetuned_sam_dir,
        device=device
    )
    
    # Compute class distribution for balanced training
    print("\nAnalyzing dataset class distribution...")
    class_counts = np.zeros(len(VehicleSegDataset.LABELS))
    
    for idx in tqdm(range(len(train_dataset)), desc="Computing class weights"):
        sample = train_dataset[idx]
        for label_idx in sample['label_indices']:
            class_counts[label_idx] += 1
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(VehicleSegDataset.LABELS) * class_counts + 1e-6)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(VehicleSegDataset.LABELS)
    
    # Extra penalty for "Others" to prevent model collapse
    others_idx = len(VehicleSegDataset.LABELS) - 1
    class_weights[others_idx] *= 0.5  # Down-weight "Others" by 50%
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass distribution and weights:")
    for i, label in enumerate(VehicleSegDataset.LABELS):
        print(f"  {label:20s}: {int(class_counts[i]):6d} samples, weight: {class_weights[i]:.4f}")
    
    # Loss & Optimizer with improved loss function
    criterion = BackgroundAwareLoss(
        num_classes=len(VehicleSegDataset.LABELS),
        others_class_idx=others_idx,
        background_weight=1.0,
        temperature=0.07,
        others_penalty=2.0,  # Penalty for predicting "Others"
        focal_gamma=2.0,  # Focal loss for hard examples
        class_weights=class_weights_tensor
    )
    print(f"\n✓ Using Enhanced BackgroundAwareLoss:")
    print(f"  - Others class index: {others_idx}")
    print(f"  - Others prediction penalty: 2.0x")
    print(f"  - Focal loss gamma: 2.0")
    print(f"  - Class weights: Enabled")
    print(f"  - Background embedding loss: Enabled")
    
    # Only optimize CLIP visual encoder (zero-shot classification, no classifier head)
    trainable_params = list(model.clip_model.visual.parameters())
    
    print(f"\n✓ Zero-shot CLIP training:")
    print(f"  - Training CLIP visual encoder only")
    print(f"  - Text encoder frozen (used for label embeddings)")
    print(f"  - Classification via cosine similarity with text features")
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Mixed Precision Training (AMP)
    scaler = None
    if args.use_amp and AMP_AVAILABLE and device.startswith('cuda'):
        scaler = GradScaler()
        print("\n✓ Mixed Precision Training (AMP) enabled")
    else:
        print("\n✗ Mixed Precision Training disabled")
    
    # Gradient Accumulation
    if args.accumulation_steps > 1:
        print(f"✓ Gradient Accumulation: {args.accumulation_steps} steps")
        print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    # Compile model disabled (incompatible with CPU operations in extract_mask_features)
    # torch.compile causes performance degradation due to numpy/cv2 operations
    if args.compile_model:
        print("✗ Model compilation disabled (incompatible with current pipeline)")
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            scaler=scaler, accumulation_steps=args.accumulation_steps
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%, "
              f"Samples: {train_metrics['total_samples']}")
        
        # Validate
        valid_metrics = validate(
            model, valid_loader, criterion, device, epoch, writer
        )
        
        print(f"Valid Loss: {valid_metrics['loss']:.4f}, "
              f"Valid Acc: {valid_metrics['accuracy']:.2f}%, "
              f"Samples: {valid_metrics['total_samples']}")
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
        }, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if valid_metrics['accuracy'] > best_val_acc:
            best_val_acc = valid_metrics['accuracy']
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_accuracy': best_val_acc,
            }, best_model_path)
            print(f"✓ New best model saved! Accuracy: {best_val_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved in: {save_dir}")
    
    writer.close()


if __name__ == '__main__':
    main()
