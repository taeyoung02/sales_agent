"""
SAM + CLIP integrated model for vehicle part segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import open_clip
from typing import List, Tuple, Dict, Optional
import numpy as np
import os

# Import fine-tuned SAM loader
from load_finetuned_sam import load_finetuned_sam


class SAMCLIPModel(nn.Module):
    """SAM과 CLIP을 결합한 모델"""
    
    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        num_classes: int = 14,  # 13 vehicle parts + Others
        freeze_sam: bool = True,
        use_finetuned_sam: bool = False,
        finetuned_sam_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.num_classes = num_classes
        
        # SAM 모델 로드
        if use_finetuned_sam and finetuned_sam_dir:
            # Load fine-tuned SAM model
            print("\n" + "="*60)
            print("Loading Fine-tuned SAM Model")
            print("="*60)
            from load_finetuned_sam import get_sam_checkpoint_path
            finetuned_checkpoint = get_sam_checkpoint_path(finetuned_sam_dir, use_best=True)
            self.sam = load_finetuned_sam(
                checkpoint_path=finetuned_checkpoint,
                sam_model_type=sam_model_type,
                device=device
            )
            print("="*60)
        else:
            # Load pretrained SAM model
            print(f"Loading pretrained SAM from: {sam_checkpoint}")
            self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.sam.to(device)
        
        if freeze_sam:
            # SAM은 freeze (마스크 생성만 사용)
            for param in self.sam.parameters():
                param.requires_grad = False
            self.sam.eval()
        
        self.sam_predictor = SamPredictor(self.sam)
        
        # CLIP 모델 로드
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            device=device
        )
        
        # CLIP의 vision encoder는 학습
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        
        # CLIP의 text encoder는 freeze
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad = False
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False
        for param in self.clip_model.ln_final.parameters():
            param.requires_grad = False
        
        # Background class embedding: 배경 관련 단어들의 평균
        # "Others" 대신 배경을 나타내는 단어들의 의미 공간 중심점 사용
        self.background_words = [
            "asphalt", "background", "road", "surroundings", 
            "environment", "outdoor", "pavement", "ground",
            "street", "scenery", "context", "setting"
        ]
        self._init_background_embedding()
        
        # Zero-shot classification using CLIP text encoder
        # Initialize label text embeddings immediately
        from dataset import VehicleSegDataset
        self._encode_labels(VehicleSegDataset.LABELS)
        
        print("✓ Using CLIP zero-shot classification (text embeddings)")
        print("  No linear classifier - using cosine similarity with text features")
    
    def _encode_labels(self, label_names: List[str]):
        """
        Encode label names into CLIP text embeddings for zero-shot classification
        
        Args:
            label_names: List of label strings (e.g., ["Back window", "Front window", ...])
        """
        self.label_names = label_names
        
        tokenizer = open_clip.get_tokenizer(self.clip_model.__class__.__name__.replace('CLIP', 'ViT-B-32'))
        
        with torch.no_grad():
            # Tokenize label names
            tokens = tokenizer(label_names).to(self.device)
            
            # Encode with CLIP text encoder
            text_features = self.clip_model.encode_text(tokens)
            
            # L2 normalize (standard CLIP)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Store as buffer (will be saved with model but not trained)
            # Use direct assignment instead of register_buffer to allow updates
            if hasattr(self, 'label_text_features'):
                # Update existing buffer
                self.label_text_features = text_features
            else:
                # Register new buffer
                self.register_buffer('label_text_features', text_features)
        
        print(f"✓ Encoded {len(label_names)} label text embeddings")
        print(f"  Labels: {', '.join(label_names[:5])}{'...' if len(label_names) > 5 else ''}")
    
    def _init_background_embedding(self):
        """
        배경 관련 단어들의 평균 임베딩을 계산하여 Others 클래스의 타겟으로 설정
        
        CLIP text encoder를 사용하여 배경 관련 단어들을 임베딩하고,
        이들의 평균을 구하여 분류되지 않은 마스크(Others)의 목표 임베딩으로 사용
        """
        tokenizer = open_clip.get_tokenizer(self.clip_model.__class__.__name__.replace('CLIP', 'ViT-B-32'))
        
        with torch.no_grad():
            # 배경 단어들을 토큰화
            tokens = tokenizer(self.background_words).to(self.device)
            
            # CLIP text encoder로 임베딩 추출
            text_features = self.clip_model.encode_text(tokens)
            
            # L2 정규화 (CLIP feature space에서 표준)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 평균 임베딩 계산
            self.background_embedding = text_features.mean(dim=0, keepdim=True)  # (1, clip_dim)
            
            # 다시 정규화 (평균 후에도 unit vector 유지)
            self.background_embedding = self.background_embedding / self.background_embedding.norm(dim=-1, keepdim=True)
        
        print(f"✓ Background embedding initialized from {len(self.background_words)} words:")
        print(f"  {', '.join(self.background_words)}")
        print(f"  Embedding shape: {self.background_embedding.shape}")
    
    def generate_sam_masks(
        self, 
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95
    ) -> List[Dict]:
        """
        SAM을 사용하여 이미지에서 마스크 생성
        
        Args:
            image: RGB image (H, W, 3)
            points_per_side: Grid 점의 수
            pred_iou_thresh: IoU threshold
            stability_score_thresh: Stability score threshold
            
        Returns:
            List of mask dictionaries
        """
        from segment_anything import SamAutomaticMaskGenerator
        
        mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        
        with torch.no_grad():
            masks = mask_generator.generate(image)
        
        return masks
    
    def extract_mask_features(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor,
        use_context_blur: bool = True,
        expand_ratio: float = 0.15,
        blur_strength: Tuple[int, int] = (51, 51)
    ) -> torch.Tensor:
        """
        마스크 영역의 CLIP 특징 추출 with Context-Aware Blur Cropping
        
        Args:
            image: (3, H, W) tensor
            mask: (H, W) binary mask tensor
            use_context_blur: If True, use context-aware blur cropping
            expand_ratio: Bbox expansion ratio for context preservation
            blur_strength: Gaussian blur kernel size for background suppression
            
        Returns:
            Feature vector
        """
        import cv2
        
        # Convert to numpy for processing
        image_np = image.cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
        image_np = (image_np * 255).astype(np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        if use_context_blur:
            # Context-Aware Blur Cropping (from preprocess.py)
            cropped_np = self._get_context_aware_crop(
                mask_np, image_np, expand_ratio, blur_strength
            )
        else:
            # Original method: tight bbox crop
            y_indices, x_indices = np.where(mask_np > 0)
            
            if len(y_indices) == 0:
                return torch.zeros(self.clip_model.visual.output_dim, device=self.device)
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            cropped_np = image_np[y_min:y_max+1, x_min:x_max+1]
        
        # Resize to 224x224
        cropped_np = cv2.resize(cropped_np, (224, 224))
        
        # Convert back to tensor
        cropped = torch.from_numpy(cropped_np).permute(2, 0, 1).float() / 255.0
        cropped = cropped.unsqueeze(0).to(self.device)
        
        # CLIP normalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        cropped = (cropped - mean) / std
        
        # CLIP feature 추출
        with torch.set_grad_enabled(self.training):
            features = self.clip_model.encode_image(cropped)
        
        return features.squeeze(0)
    
    @staticmethod
    def _get_context_aware_crop(
        mask: np.ndarray,
        image: np.ndarray,
        expand_ratio: float = 0.15,
        blur_strength: Tuple[int, int] = (51, 51)
    ) -> np.ndarray:
        """
        Context-Aware Blur Cropping (same as preprocess.py)
        
        1. Expand bbox to include context (e.g., wheel + fender)
        2. Blur background to force CLIP focus on mask region
        """
        import cv2
        
        h_img, w_img = image.shape[:2]
        
        # Get bounding box from mask
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        w, h = x_max - x_min + 1, y_max - y_min + 1
        
        # Expand bbox
        pad_w = int(w * expand_ratio)
        pad_h = int(h * expand_ratio)
        
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w_img, x_max + pad_w + 1)
        y2 = min(h_img, y_max + pad_h + 1)
        
        if x2 <= x1 or y2 <= y1:
            x1, y1 = x_min, y_min
            x2, y2 = x_max + 1, y_max + 1
        
        # Crop
        cropped_img = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()
        
        # Blur background
        try:
            blurred_img = cv2.GaussianBlur(cropped_img, blur_strength, 0)
        except cv2.error:
            blurred_img = cropped_img.copy()
        
        # Blend
        mask_3ch = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=-1).astype(np.float32)
        final_img = (cropped_img.astype(np.float32) * mask_3ch + 
                     blurred_img.astype(np.float32) * (1.0 - mask_3ch))
        
        return np.clip(final_img, 0, 255).astype(np.uint8)
    
    def forward(
        self, 
        image: torch.Tensor, 
        masks: List[torch.Tensor],
        use_context_blur: bool = True,
        expand_ratio: float = 0.15,
        blur_strength: Tuple[int, int] = (51, 51),
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Context-Aware Blur Cropping
        
        Args:
            image: (3, H, W) tensor
            masks: List of binary masks (H, W)
            use_context_blur: Enable context-aware blur cropping (default: True)
            expand_ratio: Bbox expansion ratio for context (default: 0.3)
            blur_strength: Gaussian blur kernel for background (default: (51, 51))
            return_features: If True, return (logits, features) instead of just logits
            
        Returns:
            If return_features=False: Logits (N, num_classes)
            If return_features=True: (Logits (N, num_classes), Features (N, clip_dim))
        """
        if len(masks) == 0:
            empty_logits = torch.empty(0, self.num_classes, device=self.device)
            empty_features = torch.empty(0, self.clip_model.visual.output_dim, device=self.device)
            return (empty_logits, empty_features) if return_features else empty_logits
        
        # 각 마스크에 대해 특징 추출 (Context-Aware Blur Cropping 적용)
        features = []
        for mask in masks:
            feat = self.extract_mask_features(
                image, mask, 
                use_context_blur=use_context_blur,
                expand_ratio=expand_ratio,
                blur_strength=blur_strength
            )
            features.append(feat)
        
        features = torch.stack(features)  # (N, clip_dim)
        
        # Normalize visual features
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Zero-shot classification using text embeddings
        # Compute cosine similarity with label text features
        # features: (N, clip_dim), label_text_features: (num_classes, clip_dim)
        logits = features @ self.label_text_features.T  # (N, num_classes)
        
        # Scale by temperature for better gradients (standard CLIP practice)
        temperature = 0.07
        logits = logits / temperature
        
        if return_features:
            return logits, features
        return logits
    
    def predict(
        self, 
        image: np.ndarray,
        return_masks: bool = True
    ) -> Tuple[List[str], List[float], List[np.ndarray]]:
        """
        이미지에서 마스크를 생성하고 각 마스크를 분류
        
        Args:
            image: RGB image (H, W, 3) numpy array
            return_masks: 마스크도 반환할지 여부
            
        Returns:
            (predicted_labels, confidences, masks)
        """
        self.eval()
        
        # SAM으로 마스크 생성
        sam_masks = self.generate_sam_masks(image)
        
        if len(sam_masks) == 0:
            return [], [], []
        
        # 이미지를 tensor로 변환
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # 마스크를 tensor로 변환
        mask_tensors = [
            torch.from_numpy(m['segmentation'].astype(np.float32)).to(self.device)
            for m in sam_masks
        ]
        
        # 분류
        with torch.no_grad():
            logits = self.forward(image_tensor, mask_tensors)
            probs = F.softmax(logits, dim=1)
            confidences, predicted_indices = probs.max(dim=1)
        
        # 결과 정리
        from dataset import VehicleSegDataset
        predicted_labels = [
            VehicleSegDataset.LABELS[idx.item()] 
            for idx in predicted_indices
        ]
        confidences = confidences.cpu().numpy().tolist()
        
        if return_masks:
            masks = [m['segmentation'] for m in sam_masks]
            return predicted_labels, confidences, masks
        else:
            return predicted_labels, confidences, []
