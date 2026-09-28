"""
SAM Fine-tuning Model
Only fine-tune mask decoder and prompt encoder, freeze image encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from typing import List, Tuple, Dict, Optional
import numpy as np


class SAMFineTuner(nn.Module):
    """
    SAM Fine-tuning model for vehicle part segmentation
    
    Strategy:
    - Freeze: Image Encoder (ViT) - pretrained feature extraction
    - Train: Mask Decoder - adapt to vehicle parts
    - Train: Prompt Encoder - learn better prompt representations
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        freeze_image_encoder: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        
        # Load SAM model
        self.sam: Sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        
        # Freeze image encoder (ViT backbone)
        if freeze_image_encoder:
            print("✓ Freezing SAM Image Encoder (ViT)")
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            self.sam.image_encoder.eval()
        else:
            print("✗ Image Encoder will be fine-tuned (warning: very expensive)")
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = True
        
        # Train mask decoder
        print("✓ Training SAM Mask Decoder")
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        
        # Train prompt encoder
        print("✓ Training SAM Prompt Encoder")
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True
        
        # Image preprocessing
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
        self.img_size = self.sam.image_encoder.img_size  # 1024
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for SAM
        Args:
            image: (H, W, 3) numpy array RGB [0, 255]
        Returns:
            tensor: (1, 3, 1024, 1024) normalized
        """
        # Resize to 1024x1024
        h, w = image.shape[:2]
        if h != self.img_size or w != self.img_size:
            image = torch.nn.functional.interpolate(
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        # Normalize
        image = image.to(self.device)
        image = (image - self.pixel_mean) / self.pixel_std
        
        return image
    
    def preprocess_mask(self, mask: np.ndarray, orig_size: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess ground truth mask
        Args:
            mask: (H, W) numpy array binary mask
            orig_size: (h, w) original image size
        Returns:
            tensor: (1, 1, 256, 256) resized mask for loss computation
        """
        # SAM outputs at 256x256 resolution
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        
        # Resize to 256x256 (SAM output resolution)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(256, 256),
            mode='nearest'
        )
        
        return mask_tensor.to(self.device)
    
    def forward(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prompts
        
        Args:
            image: (H, W, 3) numpy array RGB
            bbox: (4,) [x1, y1, x2, y2] or None
            points: (N, 2) [x, y] coordinates or None
            point_labels: (N,) 1 for foreground, 0 for background
            return_logits: Return logits instead of probabilities
            
        Returns:
            masks: (1, 1, 256, 256) predicted mask logits or probabilities
            iou_predictions: (1,) predicted IoU score
        """
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image_tensor)
        
        # Prepare prompts
        prompt_points = None
        prompt_labels = None
        prompt_boxes = None
        
        if points is not None and point_labels is not None:
            # Scale points to 1024x1024
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            prompt_points = torch.from_numpy(points).float().to(self.device)
            prompt_points = prompt_points * torch.tensor([scale_x, scale_y]).to(self.device)
            prompt_points = prompt_points.unsqueeze(0)  # (1, N, 2)
            
            prompt_labels = torch.from_numpy(point_labels).float().to(self.device)
            prompt_labels = prompt_labels.unsqueeze(0)  # (1, N)
        
        if bbox is not None:
            # Scale bbox to 1024x1024
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            prompt_boxes = torch.from_numpy(bbox).float().to(self.device)
            prompt_boxes = prompt_boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y]).to(self.device)
            prompt_boxes = prompt_boxes.unsqueeze(0)  # (1, 4)
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(prompt_points, prompt_labels) if prompt_points is not None else None,
            boxes=prompt_boxes,
            masks=None,
        )
        
        # Predict masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,  # Single mask output
        )
        
        # low_res_masks: (1, 1, 256, 256)
        if not return_logits:
            low_res_masks = torch.sigmoid(low_res_masks)
        
        return low_res_masks, iou_predictions
    
    def forward_with_embeddings(
        self,
        image_embeddings: torch.Tensor,
        orig_size: Tuple[int, int],
        bbox: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with PRE-COMPUTED image embeddings (MUCH FASTER!)
        
        This avoids re-encoding the image for each mask, which is the biggest bottleneck.
        Image encoding takes ~80% of forward pass time.
        
        Args:
            image_embeddings: (1, 256, 64, 64) pre-computed from image_encoder
            orig_size: (h, w) original image size
            bbox: (4,) [x1, y1, x2, y2] or None
            points: (N, 2) [x, y] coordinates or None
            point_labels: (N,) 1 for foreground, 0 for background
            return_logits: Return logits instead of probabilities
            
        Returns:
            masks: (1, 1, 256, 256) predicted mask logits or probabilities
            iou_predictions: (1,) predicted IoU score
        """
        orig_h, orig_w = orig_size
        
        # Prepare prompts
        prompt_points = None
        prompt_labels = None
        prompt_boxes = None
        
        if points is not None and point_labels is not None:
            # Scale points to 1024x1024
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            prompt_points = torch.from_numpy(points).float().to(self.device)
            prompt_points = prompt_points * torch.tensor([scale_x, scale_y]).to(self.device)
            prompt_points = prompt_points.unsqueeze(0)  # (1, N, 2)
            
            prompt_labels = torch.from_numpy(point_labels).float().to(self.device)
            prompt_labels = prompt_labels.unsqueeze(0)  # (1, N)
        
        if bbox is not None:
            # Scale bbox to 1024x1024
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            prompt_boxes = torch.from_numpy(bbox).float().to(self.device)
            prompt_boxes = prompt_boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y]).to(self.device)
            prompt_boxes = prompt_boxes.unsqueeze(0)  # (1, 4)
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(prompt_points, prompt_labels) if prompt_points is not None else None,
            boxes=prompt_boxes,
            masks=None,
        )
        
        # Predict masks (using pre-computed image_embeddings!)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        if not return_logits:
            low_res_masks = torch.sigmoid(low_res_masks)
        
        return low_res_masks, iou_predictions
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (mask decoder + prompt encoder)"""
        trainable_params = []
        trainable_params += list(self.sam.mask_decoder.parameters())
        trainable_params += list(self.sam.prompt_encoder.parameters())
        return trainable_params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component"""
        total_params = sum(p.numel() for p in self.sam.parameters())
        trainable_params = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.sam.image_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.sam.mask_decoder.parameters())
        prompt_params = sum(p.numel() for p in self.sam.prompt_encoder.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'image_encoder': encoder_params,
            'mask_decoder': decoder_params,
            'prompt_encoder': prompt_params,
        }


if __name__ == "__main__":
    # Test model
    print("Testing SAMFineTuner...")
    
    model = SAMFineTuner(
        sam_checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
        freeze_image_encoder=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,} ({count/1e6:.2f}M)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = np.array([100, 100, 300, 300], dtype=np.float32)
    dummy_points = np.array([[200, 200], [250, 250]], dtype=np.float32)
    dummy_labels = np.array([1, 1], dtype=np.float32)
    
    with torch.no_grad():
        masks, iou_pred = model(
            dummy_image,
            bbox=dummy_bbox,
            points=dummy_points,
            point_labels=dummy_labels
        )
    
    print(f"  Output mask shape: {masks.shape}")
    print(f"  IoU prediction: {iou_pred.item():.4f}")
    print("\n✓ SAMFineTuner test passed!")
