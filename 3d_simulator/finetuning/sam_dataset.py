"""
SAM Fine-tuning Dataset for VehicleSeg10K
Converts polygon annotations to binary masks for SAM training
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import random


class SAMDataset(Dataset):
    """VehicleSeg10K dataset for SAM fine-tuning with prompt-based learning"""
    
    # 13 vehicle part labels (excluding "Others" background class)
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
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_prompts_per_mask: int = 5,
        use_bbox_prompt: bool = True,
        use_point_prompt: bool = True,
        augment: bool = True
    ):
        """
        Args:
            root_dir: VehicleSeg10K 데이터셋 루트 디렉토리
            split: 'train' or 'valid'
            num_prompts_per_mask: 각 마스크마다 생성할 프롬프트 수
            use_bbox_prompt: Bounding box 프롬프트 사용 여부
            use_point_prompt: Point 프롬프트 사용 여부
            augment: 데이터 증강 여부 (horizontal flip)
        """
        self.root_dir = root_dir
        self.split = split
        self.num_prompts_per_mask = 1  # [OPTIMIZATION] Only generate 1 prompt (was num_prompts_per_mask)
        self.use_bbox_prompt = use_bbox_prompt
        self.use_point_prompt = use_point_prompt
        self.augment = augment and (split == 'train')
        
        self.images_dir = os.path.join(root_dir, "images", split)
        self.annotations_dir = os.path.join(root_dir, "annotations", split)
        
        # 이미지 파일 리스트
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith(('.jpg', '.png'))
        ])
        
        print(f"SAMDataset ({split}): {len(self.image_files)} images")
        print(f"  - Prompts per mask: {num_prompts_per_mask}")
        print(f"  - Use bbox prompt: {use_bbox_prompt}")
        print(f"  - Use point prompt: {use_point_prompt}")
        print(f"  - Augmentation: {self.augment}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _polygon_to_mask(self, points: List[List[float]], h: int, w: int) -> np.ndarray:
        """Convert polygon points to binary mask"""
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
        return mask
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from binary mask (x1, y1, x2, y2)"""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return 0, 0, 0, 0
        
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        return int(x1), int(y1), int(x2), int(y2)
    
    def _sample_points_from_mask(self, mask: np.ndarray, num_points: int = 1) -> np.ndarray:
        """Sample random points from inside the mask
        Returns: (num_points, 2) array of [x, y] coordinates
        """
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.zeros((num_points, 2), dtype=np.float32)
        
        # Randomly sample points
        indices = np.random.choice(len(ys), size=min(num_points, len(ys)), replace=False)
        points = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
        
        # If we need more points than available, repeat sampling
        if len(points) < num_points:
            points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='edge')
        
        return points
    
    def _apply_horizontal_flip(self, image: np.ndarray, masks: List[np.ndarray], 
                               label_names: List[str]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """Apply horizontal flip to image and masks, update left/right labels"""
        image = np.fliplr(image).copy()
        masks = [np.fliplr(mask).copy() for mask in masks]
        
        # Swap left/right labels
        flipped_labels = []
        for label in label_names:
            if "Left" in label:
                flipped_labels.append(label.replace("Left", "Right"))
            elif "Right" in label:
                flipped_labels.append(label.replace("Right", "Left"))
            else:
                flipped_labels.append(label)
        
        return image, masks, flipped_labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - image: (H, W, 3) numpy array RGB
                - masks: List[(H, W)] binary masks
                - labels: List[str] label names
                - prompts: List[Dict] prompts for each mask
                    - bbox: (4,) [x1, y1, x2, y2] or None
                    - points: (N, 2) [x, y] coordinates or None
                    - point_labels: (N,) 1 for foreground, 0 for background
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Load annotation
        ann_name = img_name.rsplit('.', 1)[0] + '.json'
        ann_path = os.path.join(self.annotations_dir, ann_name)
        
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        # Convert polygons to masks
        masks = []
        labels = []
        
        for shape in annotation['shapes']:
            label = shape['label']
            
            # Skip "Others" background class
            if label not in self.LABELS:
                continue
            
            points = shape['points']
            
            # Convert polygon to mask
            mask = self._polygon_to_mask(points, h, w)
            
            # Only keep masks with valid area
            if mask.sum() > 0:
                masks.append(mask)
                labels.append(label)
        
        # Apply augmentation
        if self.augment and random.random() < 0.5:
            img_array, masks, labels = self._apply_horizontal_flip(img_array, masks, labels)
        
        # Generate prompts for each mask
        prompts = []
        for mask in masks:
            prompt_list = []
            
            for _ in range(self.num_prompts_per_mask):
                prompt = {}
                
                # Random choice: bbox, point, or both
                use_bbox = self.use_bbox_prompt and random.random() < 0.5
                use_point = self.use_point_prompt and random.random() < 0.5
                
                # Ensure at least one prompt type is used
                if not use_bbox and not use_point:
                    if random.random() < 0.5:
                        use_bbox = True
                    else:
                        use_point = True
                
                if use_bbox:
                    bbox = self._get_bbox_from_mask(mask)
                    # Add small random noise to bbox (±5 pixels)
                    noise = np.random.randint(-5, 6, size=4)
                    bbox = np.array(bbox) + noise
                    bbox = np.clip(bbox, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
                    prompt['bbox'] = bbox.astype(np.float32)
                else:
                    prompt['bbox'] = None
                
                if use_point:
                    # Sample 1-5 positive points
                    num_pos_points = random.randint(1, 5)
                    pos_points = self._sample_points_from_mask(mask, num_pos_points)
                    
                    # Optionally add negative points (outside mask)
                    num_neg_points = random.randint(0, 2)
                    if num_neg_points > 0:
                        neg_mask = 1 - mask
                        neg_points = self._sample_points_from_mask(neg_mask, num_neg_points)
                        
                        points = np.vstack([pos_points, neg_points])
                        point_labels = np.array([1] * num_pos_points + [0] * num_neg_points, dtype=np.float32)
                    else:
                        points = pos_points
                        point_labels = np.ones(num_pos_points, dtype=np.float32)
                    
                    prompt['points'] = points
                    prompt['point_labels'] = point_labels
                else:
                    prompt['points'] = None
                    prompt['point_labels'] = None
                
                prompt_list.append(prompt)
            
            prompts.append(prompt_list)
        
        return {
            'image': img_array,
            'masks': masks,
            'labels': labels,
            'prompts': prompts,
            'image_name': img_name
        }


def sam_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    SAM은 배치 단위 학습이 어려우므로 리스트 그대로 반환
    각 이미지마다 크기와 마스크 개수가 다르기 때문
    """
    return batch


if __name__ == "__main__":
    # Test dataset
    dataset = SAMDataset(
        root_dir="VehicleSeg10K",
        split="train",
        num_prompts_per_mask=5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Num masks: {len(sample['masks'])}")
    print(f"  Labels: {sample['labels']}")
    print(f"  Num prompts per mask: {len(sample['prompts'][0]) if sample['prompts'] else 0}")
    
    if sample['prompts']:
        print(f"\n  First mask prompts:")
        for i, prompt in enumerate(sample['prompts'][0][:2]):  # Show first 2 prompts
            print(f"    Prompt {i+1}:")
            if prompt['bbox'] is not None:
                print(f"      BBox: {prompt['bbox']}")
            if prompt['points'] is not None:
                print(f"      Points: {prompt['points'].shape}")
                print(f"      Point labels: {prompt['point_labels']}")
