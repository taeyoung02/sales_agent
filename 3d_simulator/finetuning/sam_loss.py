"""
Loss functions for SAM fine-tuning
Combines Focal Loss (hard examples) + Dice Loss (boundary precision)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples (misclassified samples)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, 1, H, W) predicted logits
            targets: (B, 1, H, W) ground truth masks [0, 1]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Focal weight: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Better for handling boundary precision and class imbalance
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, 1, H, W) predicted logits
            targets: (B, 1, H, W) ground truth masks [0, 1]
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        # Dice loss
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss
        else:
            return dice_loss


class SAMSegmentationLoss(nn.Module):
    """
    Combined loss for SAM fine-tuning:
    - Focal Loss: Handle hard examples and class imbalance
    - Dice Loss: Improve boundary precision
    - IoU Loss: Align with IoU metric
    """
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        iou_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_iou: torch.Tensor
    ) -> dict:
        """
        Args:
            pred_masks: (B, 1, H, W) predicted mask logits
            gt_masks: (B, 1, H, W) ground truth masks [0, 1]
            pred_iou: (B,) predicted IoU scores
            
        Returns:
            dict with 'total', 'focal', 'dice', 'iou' losses
        """
        # Focal loss
        focal = self.focal_loss(pred_masks, gt_masks)
        
        # Dice loss
        dice = self.dice_loss(pred_masks, gt_masks)
        
        # IoU loss (MSE between predicted IoU and actual IoU)
        with torch.no_grad():
            # Calculate actual IoU
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            intersection = (pred_binary * gt_masks).sum(dim=[1, 2, 3])
            union = pred_binary.sum(dim=[1, 2, 3]) + gt_masks.sum(dim=[1, 2, 3]) - intersection
            actual_iou = (intersection + 1e-6) / (union + 1e-6)
        
        iou_loss = F.mse_loss(pred_iou.squeeze(), actual_iou)
        
        # Total loss
        total = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.iou_weight * iou_loss
        )
        
        return {
            'total': total,
            'focal': focal.item(),
            'dice': dice.item(),
            'iou': iou_loss.item()
        }


if __name__ == "__main__":
    # Test losses
    print("Testing SAM loss functions...")
    
    # Dummy data
    B, H, W = 4, 256, 256
    pred_masks = torch.randn(B, 1, H, W)  # Logits
    gt_masks = torch.randint(0, 2, (B, 1, H, W)).float()
    pred_iou = torch.rand(B)
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    focal = focal_loss(pred_masks, gt_masks)
    print(f"Focal Loss: {focal.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice = dice_loss(pred_masks, gt_masks)
    print(f"Dice Loss: {dice.item():.4f}")
    
    # Test Combined Loss
    combined_loss = SAMSegmentationLoss()
    losses = combined_loss(pred_masks, gt_masks, pred_iou)
    print(f"\nCombined Loss:")
    for name, value in losses.items():
        if name == 'total':
            print(f"  {name}: {value.item():.4f}")
        else:
            print(f"  {name}: {value:.4f}")
    
    print("\n✓ All loss functions tested successfully!")
