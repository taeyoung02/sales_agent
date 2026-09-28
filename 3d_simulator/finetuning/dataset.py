"""
VehicleSeg10K Dataset for SAM + CLIP fine-tuning
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class VehicleSegDataset(Dataset):
    """VehicleSeg10K dataset for vehicle part segmentation"""
    
    # VehicleSeg10K 데이터셋의 실제 라벨 (JSON 파일에서 사용되는 원본 라벨)
    LABELS = [
        "Back window",
        "Foreground",
        "Front window",
        "Left back door",
        "Left back window",
        "Left front door",
        "Left front window",
        "Plate",
        "Right back door",
        "Right back window",
        "Right front door",
        "Right front window",
        "Wheel",
        "Others"  # SAM이 찾은 마스크가 위 라벨에 해당하지 않을 때
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        iou_threshold: float = 0.5  # Ground truth와 매칭할 때 사용할 IoU threshold
    ):
        """
        Args:
            root_dir: VehicleSeg10K 데이터셋 루트 디렉토리
            split: 'train' or 'valid'
            transform: 이미지 변환
            iou_threshold: GT 마스크와 SAM 마스크 매칭을 위한 IoU threshold
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.iou_threshold = iou_threshold
        
        self.images_dir = os.path.join(root_dir, "images", split)
        self.annotations_dir = os.path.join(root_dir, "annotations", split)
        
        # 이미지 파일 리스트
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith(('.jpg', '.png'))
        ])
        
        # 라벨을 인덱스로 매핑
        self.label_to_idx = {label: idx for idx, label in enumerate(self.LABELS)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.LABELS)}
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - image: PIL Image
                - image_path: str
                - masks: list of binary masks (H, W) - includes "Others" for non-annotated regions
                - labels: list of label names
                - label_indices: list of label indices
                - annotation: original annotation dict
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Annotation 로드
        ann_name = img_name.rsplit('.', 1)[0] + '.json'
        ann_path = os.path.join(self.annotations_dir, ann_name)
        
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        # Polygon을 마스크로 변환
        masks = []
        labels = []
        annotated_mask = np.zeros((h, w), dtype=np.uint8)  # 전체 annotation 영역 추적
        
        for shape in annotation.get('shapes', []):
            label = shape.get('label', 'Others')
            points = shape.get('points', [])
            
            if len(points) < 3:  # polygon은 최소 3개 점 필요
                continue
            
            # Polygon을 binary mask로 변환
            mask = self._polygon_to_mask(points, h, w)
            
            masks.append(mask)
            labels.append(label)
            
            # Annotated 영역에 추가
            annotated_mask = np.logical_or(annotated_mask, mask).astype(np.uint8)
        
        # Complement: annotation되지 않은 영역을 "Others" (배경)로 추가
        background_mask = (1 - annotated_mask).astype(np.uint8)
        if background_mask.sum() > 0:  # 배경 영역이 존재하면
            masks.append(background_mask)
            labels.append("Others")
        
        # 라벨을 인덱스로 변환
        label_indices = [
            self.label_to_idx.get(label, self.label_to_idx["Others"]) 
            for label in labels
        ]
        
        sample = {
            'image': image,
            'image_path': img_path,
            'masks': masks,
            'labels': labels,
            'label_indices': label_indices,
            'annotation': annotation,
            'image_array': img_array
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _polygon_to_mask(self, points: List[List[float]], h: int, w: int) -> np.ndarray:
        """폴리곤 좌표를 binary mask로 변환"""
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
        return mask
    
    @staticmethod
    def get_context_aware_crop(
        mask: np.ndarray,
        image: np.ndarray,
        expand_ratio: float = 0.15,
        blur_strength: Tuple[int, int] = (51, 51)
    ) -> np.ndarray:
        """
        Context-Aware Blur Cropping Strategy:
        
        Problems with tight bbox cropping:
        1. Loss of Context: CLIP can't tell if it's a "door handle" or just "metal cylinder"
        2. Background Noise: Random background elements distract CLIP's attention
        
        Solution:
        1. Expand bbox to include context (e.g., wheel + fender)
        2. Blur background within crop to force CLIP focus on the mask region
        
        Args:
            mask: Binary mask (H x W)
            image: RGB image (H x W x 3)
            expand_ratio: How much to expand bbox (0.2-0.4 recommended)
            blur_strength: Gaussian blur kernel size (must be odd numbers)
        
        Returns:
            Cropped image with contextual background blurred
        """
        h_img, w_img = image.shape[:2]
        
        # Get bounding box from mask
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            # Empty mask: return small blank image
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        w, h = x_max - x_min + 1, y_max - y_min + 1
        
        # 1. Expand Bounding Box (Context Preservation)
        pad_w = int(w * expand_ratio)
        pad_h = int(h * expand_ratio)
        
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w_img, x_max + pad_w + 1)
        y2 = min(h_img, y_max + pad_h + 1)
        
        # Safety check
        if x2 <= x1 or y2 <= y1:
            x1, y1 = x_min, y_min
            x2, y2 = x_max + 1, y_max + 1
        
        # Crop image and mask
        cropped_img = image[y1:y2, x1:x2, ...].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()
        
        # 2. Blur Masking (Focus Enhancement)
        try:
            blurred_img = cv2.GaussianBlur(cropped_img, blur_strength, 0)
        except cv2.error:
            # Fallback if blur fails
            blurred_img = cropped_img.copy()
        
        # Convert mask to 3-channel for blending
        mask_3ch = np.stack([cropped_mask, cropped_mask, cropped_mask], axis=-1).astype(np.float32)
        
        # Blend: Sharp object + Blurred background
        final_img = (cropped_img.astype(np.float32) * mask_3ch + 
                     blurred_img.astype(np.float32) * (1.0 - mask_3ch))
        
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        return final_img
    
    @staticmethod
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """두 마스크 간의 IoU 계산"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_sam_mask_to_gt(
        self, 
        sam_mask: np.ndarray, 
        gt_masks: List[np.ndarray], 
        gt_labels: List[str]
    ) -> Tuple[str, float]:
        """
        SAM이 생성한 마스크를 GT 마스크와 매칭하여 라벨 결정
        
        Args:
            sam_mask: SAM이 생성한 binary mask
            gt_masks: Ground truth masks 리스트
            gt_labels: Ground truth labels 리스트
            
        Returns:
            (matched_label, max_iou)
        """
        max_iou = 0.0
        matched_label = "Others"
        
        for gt_mask, gt_label in zip(gt_masks, gt_labels):
            iou = self.compute_iou(sam_mask, gt_mask)
            
            if iou > max_iou:
                max_iou = iou
                matched_label = gt_label
        
        # IoU threshold 이하면 "Others"로 분류
        if max_iou < self.iou_threshold:
            matched_label = "Others"
        
        return matched_label, max_iou


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # 배치의 각 샘플을 리스트로 반환 (이미지 크기가 다를 수 있음)
    return batch
