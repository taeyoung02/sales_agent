import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

try:
    import timm
except ImportError:
    print("Warning: timm is not installed. DINOv2 features will not be available.")
    print("Install it with: pip install timm")


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-32"
    clip_model_pretrained: str = "openai"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

@dataclass
class DINOv2NetworkConfig:
    _target: Type = field(default_factory=lambda: DINOv2Network)
    model_name: str = "vit_base_patch14_dinov2.lvd142m"  # or vit_small, vit_large, vit_giant
    dino_n_dims: int = 768  # 768 for base, 384 for small, 1024 for large, 1536 for giant

class DINOv2Network(nn.Module):
    """DINOv2 Feature Extractor for structural/spatial features"""
    def __init__(self, config: DINOv2NetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled GPU.")
        
        print(f"Loading DINOv2 model: {self.config.model_name}")
        try:
            # Load pretrained DINOv2 from timm with dynamic image size support
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                dynamic_img_size=True,  # Allow different input sizes
                img_size=224,  # Set expected input size to 224x224
            )
            self.model.eval()
            self.model = self.model.to("cuda")
            # Convert model to half precision to match input
            self.model = self.model.half()
            print("✓ DINOv2 model loaded successfully")
        except Exception as e:
            print(f"Failed to load DINOv2: {e}")
            raise
        
        self.dino_n_dims = self.config.dino_n_dims
    
    def encode_image(self, input):
        """Extract DINOv2 features from images"""
        processed_input = self.process(input).half()
        with torch.no_grad():
            features = self.model.forward_features(processed_input)
            # Get CLS token or average pool patch tokens
            if hasattr(features, 'shape') and len(features.shape) == 3:
                # features: [B, num_patches+1, dim]
                features = features[:, 0]  # Use CLS token
            elif isinstance(features, dict):
                features = features['x_norm_clstoken']  # For newer timm versions
        return features

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig, checkpoint_path=None):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled GPU.")
        
        # Load custom checkpoint if provided, otherwise download pretrained model
        if checkpoint_path is not None:
            print(f"✓ Local CLIP checkpoint detected: {checkpoint_path}")
            print("  Skipping pretrained model download, loading from local checkpoint...")
            
            # Create model without pretrained weights
            model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model_type,  # e.g., ViT-B-32
                pretrained=None,  # Don't download pretrained weights
                precision="fp16",
            )
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Convert MPS tensors to CPU/CUDA compatible format
            converted_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    # Move from MPS to CPU first, then will move to CUDA
                    if value.device.type == 'mps':
                        value = value.cpu()
                converted_state_dict[key] = value
            
            model.load_state_dict(converted_state_dict, strict=False)
            print("✓ Local CLIP checkpoint loaded successfully")
        else:
            print(f"No local CLIP checkpoint found, downloading pretrained model: {self.config.clip_model_pretrained}")
            model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model_type,  # e.g., ViT-B-32
                pretrained=self.config.clip_model_pretrained,  # e.g., openai
                precision="fp16",
            )
            print("✓ Pretrained CLIP model downloaded and loaded")
        
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)





def create(image_list, data_list, save_folder, use_dinov2=False, use_mask_aggregation=True):
    assert image_list is not None, "image_list must be provided to generate features"
    
    # Feature dimension calculation - CLIP only (512 dims)
    clip_dim = 512
    embed_size = clip_dim  # CLIP only, no DINOv2
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Mode:")
    print(f"  - CLIP only: {embed_size} dimensions")
    print(f"  - Context-Aware Blur Cropping: {'Enabled' if use_mask_aggregation else 'Disabled'}")
    print(f"{'='*60}\n")
    
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(
                img.unsqueeze(0), 
                sam_encoder,
                use_dinov2=False,  # Disable DINOv2
                use_mask_aggregation=use_mask_aggregation,
                original_image=img  # Pass original image for potential mask aggregation
            )
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map_stacked = torch.stack(seg_map_tensor, dim=0)
        
        # Pad to 4 channels if needed (default, s, m, l)
        num_levels = seg_map_stacked.shape[0]
        if num_levels < 4:
            padding = torch.full((4 - num_levels, *seg_map_stacked.shape[1:]), -1, dtype=seg_map_stacked.dtype)
            seg_map_stacked = torch.cat([seg_map_stacked, padding], dim=0)
        
        seg_maps[i] = seg_map_stacked

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder, use_dinov2=False, use_mask_aggregation=True, original_image=None):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs, use_mask_aggregation=use_mask_aggregation)

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        if mode not in seg_images:
            continue
        tiles = seg_images[mode]
        
        # Skip encoding if there are no tiles (empty masks)
        if tiles.shape[0] == 0:
            # CLIP only: 512 dimensions
            clip_embeds[mode] = torch.zeros((0, 512), dtype=torch.float16)
            continue
            
        tiles = tiles.to("cuda")
        
        # Extract CLIP features only
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        
        # Use CLIP features only (no DINOv2)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def get_cropped_seg_img(mask, image):
    """
    DEPRECATED: Use get_context_aware_crop instead for better feature quality.
    
    Mask-CLIP Aggregation: Crop the mask region for high-resolution CLIP encoding.
    Instead of encoding the entire image, we crop each mask's bounding box region,
    which allows CLIP to focus on individual parts without resolution loss.
    """
    x, y, w, h = np.int32(mask['bbox'])
    # Crop the region (no background masking, preserve original pixels)
    cropped_img = image[y:y+h, x:x+w, ...].copy()
    return cropped_img

def get_context_aware_crop(mask, image, expand_ratio=0.10, blur_strength=(51, 51)):
    """
    Context-Aware Blur Cropping Strategy:
    
    Problems with tight bbox cropping:
    1. Loss of Context: CLIP can't tell if it's a "door handle" or just "metal cylinder"
    2. Background Noise: Random background elements distract CLIP's attention
    
    Solution:
    1. Expand bbox to include context (e.g., wheel + fender)
    2. Blur background within crop to force CLIP focus on the mask region
    
    Args:
        mask: SAM mask dict with 'bbox' and 'segmentation' keys
        image: RGB image (numpy array, H x W x 3)
        expand_ratio: How much to expand bbox (0.2-0.4 recommended)
        blur_strength: Gaussian blur kernel size (must be odd numbers)
    
    Returns:
        Cropped image with contextual background blurred
    """
    # Validate inputs
    if 'bbox' not in mask or 'segmentation' not in mask:
        raise ValueError("Mask must contain 'bbox' and 'segmentation' keys")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be H x W x 3, got shape {image.shape}")
    
    h_img, w_img, _ = image.shape
    x, y, w, h = np.int32(mask['bbox'])
    
    # Validate bbox coordinates
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox coordinates: x={x}, y={y}, w={w}, h={h}")
    
    # 1. Expand Bounding Box (Context Preservation)
    # Extend bbox to show "where this part is attached"
    pad_w = int(w * expand_ratio)
    pad_h = int(h * expand_ratio)
    
    # Clamp to image boundaries to prevent index errors
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    
    # Safety check: ensure valid crop region
    if x2 <= x1 or y2 <= y1:
        # Fallback to original bbox if expansion fails
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    
    # Crop image
    cropped_img = image[y1:y2, x1:x2, ...].copy()
    
    # Crop mask to same region (convert global coordinates to local)
    full_mask = mask['segmentation']
    
    # Validate mask shape matches image
    if full_mask.shape[0] != h_img or full_mask.shape[1] != w_img:
        raise ValueError(f"Mask shape {full_mask.shape} doesn't match image shape {(h_img, w_img)}")
    
    cropped_mask = full_mask[y1:y2, x1:x2]
    
    # Safety check: ensure mask and cropped image have same dimensions
    if cropped_mask.shape[0] != cropped_img.shape[0] or cropped_mask.shape[1] != cropped_img.shape[1]:
        raise ValueError(f"Cropped mask shape {cropped_mask.shape} doesn't match cropped image shape {cropped_img.shape[:2]}")
    
    # 2. Blur Masking (Focus Enhancement)
    # Apply Gaussian blur to entire crop
    try:
        blurred_img = cv2.GaussianBlur(cropped_img, blur_strength, 0)
    except cv2.error as e:
        # If blur fails (e.g., image too small), use original
        print(f"Warning: Gaussian blur failed for crop size {cropped_img.shape}, using original image")
        blurred_img = cropped_img.copy()
    
    # Convert boolean mask to 3-channel float for blending
    # Ensure mask is boolean type
    mask_bool = cropped_mask.astype(bool)
    mask_3ch = np.stack([mask_bool, mask_bool, mask_bool], axis=-1).astype(np.float32)
    
    # Blend: Sharp object + Blurred background
    # final = (sharp * mask) + (blur * (1 - mask))
    final_img = (cropped_img.astype(np.float32) * mask_3ch + 
                 blurred_img.astype(np.float32) * (1.0 - mask_3ch))
    
    # Convert back to uint8 and clamp values
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_l[index] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(masks_default, masks_s, masks_m, masks_l, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = []
    
    for masks_lvl in [masks_default, masks_s, masks_m, masks_l]:
        if len(masks_lvl) == 0:
            # Keep empty lists as is
            masks_new.append(masks_lvl)
            continue
            
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new.append(masks_lvl)
    
    return tuple(masks_new)

def sam_encoder(image, use_mask_aggregation=True):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks - custom SAM returns 4 mask levels
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image, use_crop=False):
        """
        Generate segmentation maps with optional Mask-CLIP Aggregation.
        
        Args:
            masks: List of SAM masks
            image: Input image (RGB numpy array)
            use_crop: If True, use context-aware cropping for better CLIP encoding
        """
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        
        # Handle empty masks
        if len(masks) == 0:
            # Return empty tensor with correct shape (0, 3, 224, 224)
            seg_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32).to('cuda')
            return seg_imgs, seg_map
        
        for i in range(len(masks)):
            mask = masks[i]
            
            # Validate mask structure to prevent attribute errors
            if 'segmentation' not in mask or 'bbox' not in mask:
                print(f"Warning: Mask {i} missing required keys, skipping...")
                continue
            
            try:
                if use_crop:
                    # [IMPROVED] Context-Aware Blur Cropping
                    # Expand bbox to include context, blur background to enhance focus
                    seg_img = get_context_aware_crop(
                        mask, 
                        image, 
                        expand_ratio=0.3,  # 30% expansion for context
                        blur_strength=(51, 51)  # Strong blur for background suppression
                    )
                else:
                    # Original method: Mask out background (black background)
                    seg_img = get_seg_img(mask, image)
            except Exception as e:
                print(f"Warning: Failed to process mask {i}: {e}")
                print(f"  Mask bbox: {mask.get('bbox', 'N/A')}")
                print(f"  Image shape: {image.shape}")
                # Fallback to simple masking
                seg_img = get_seg_img(mask, image)
            
            # Resize and pad to 224x224 for CLIP
            try:
                pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
                seg_img_list.append(pad_seg_img)
                
                # Update segmentation map
                seg_map[masks[i]['segmentation']] = i
            except Exception as e:
                print(f"Warning: Failed to resize mask {i}: {e}, skipping...")
                continue
        
        # Handle case where all masks failed to process
        if len(seg_img_list) == 0:
            seg_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32).to('cuda')
            return seg_imgs, seg_map
        
        # Stack and convert to tensor
        seg_imgs = np.stack(seg_img_list, axis=0)  # (N, H, W, 3)
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    # Always process all mask levels, even if empty
    # Use Mask-CLIP Aggregation (cropping) for better feature extraction
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image, use_crop=use_mask_aggregation)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image, use_crop=use_mask_aggregation)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image, use_crop=use_mask_aggregation)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image, use_crop=use_mask_aggregation)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def convert_mps_to_cuda(checkpoint):
    """
    Convert checkpoint tensors from MPS (Apple Silicon) to CUDA-compatible format.
    
    Args:
        checkpoint: Model checkpoint (dict or state_dict)
    
    Returns:
        Converted checkpoint with all tensors moved from MPS to CPU
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    
    converted = {}
    mps_found = False
    
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            if value.device.type == 'mps':
                if not mps_found:
                    print("⚠️  Detected MPS tensors in checkpoint. Converting to CUDA-compatible format...")
                    mps_found = True
                value = value.cpu()
        elif isinstance(value, dict):
            # Recursively convert nested dicts
            value = convert_mps_to_cuda(value)
        converted[key] = value
    
    if mps_found:
        print("✓ MPS to CUDA conversion completed")
    
    return converted

def load_sam_checkpoint(checkpoint_path, model_type):
    """
    Load SAM checkpoint with MPS to CUDA conversion support.
    
    Args:
        checkpoint_path: Path to SAM checkpoint file
        model_type: SAM model type (vit_h, vit_b, vit_l)
    
    Returns:
        Loaded SAM model on CUDA
    """
    print(f"Loading SAM checkpoint from: {checkpoint_path}")
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Convert MPS tensors if present
    checkpoint = convert_mps_to_cuda(checkpoint)
    
    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=None)
    
    # Load state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            sam.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            sam.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is already a state dict
            sam.load_state_dict(checkpoint)
    else:
        sam.load_state_dict(checkpoint)
    
    return sam


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--no_resize', action='store_true',
                        help='Keep original image size without resizing')
    parser.add_argument('--sam_ckpt_path', type=str, default=None, 
                        help="Path to SAM checkpoint file or directory containing checkpoints. "
                             "If directory, will use the first SAM checkpoint found. "
                             "Model type (vit_h/vit_b/vit_l) auto-detected from filename.")
    parser.add_argument('--sam_checkpoint_dir', type=str, default="/workspace/data/checkpoints",
                        help="Default directory to search for SAM checkpoints if --sam_ckpt_path not provided")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    
    # Auto-detect SAM checkpoint and model type
    sam_ckpt_path = args.sam_ckpt_path
    
    # If no checkpoint specified, search in default directory
    if sam_ckpt_path is None:
        # Priority 1: Fine-tuned SAM from outputs
        finetuned_sam_paths = [
            "./outputs/sam_20260101_003941/best_model.pth",  # Local path
            "/workspace/finetuning/outputs/sam_20260101_003941/best_model.pth",  # Docker path
        ]
        
        for finetuned_sam_path in finetuned_sam_paths:
            if os.path.isfile(finetuned_sam_path):
                sam_ckpt_path = finetuned_sam_path
                print(f"✓ Found fine-tuned SAM checkpoint: {sam_ckpt_path}")
                print(f"  Training: outputs/sam_20260101_003941")
                break
        
        # Priority 2: Default checkpoint directory
        if sam_ckpt_path is None:
            checkpoint_dir = args.sam_checkpoint_dir
            if os.path.isdir(checkpoint_dir):
                # Find first SAM checkpoint file
                sam_files = [f for f in os.listdir(checkpoint_dir) 
                            if f.startswith('sam_vit_') and f.endswith('.pth')]
                if sam_files:
                    sam_files.sort()  # Deterministic order
                    sam_ckpt_path = os.path.join(checkpoint_dir, sam_files[0])
                    print(f"Auto-detected SAM checkpoint: {sam_ckpt_path}")
                    print(f"  ⚠️  Warning: Using pretrained SAM. For best results, use fine-tuned model.")
                else:
                    raise FileNotFoundError(f"No SAM checkpoint found in {checkpoint_dir}. "
                                          f"Please download a SAM model (sam_vit_h/b/l_*.pth)")
            else:
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # If checkpoint path is a directory, find first SAM checkpoint in it
    elif os.path.isdir(sam_ckpt_path):
        sam_files = [f for f in os.listdir(sam_ckpt_path) 
                    if f.startswith('sam_vit_') and f.endswith('.pth')]
        if sam_files:
            sam_files.sort()
            sam_ckpt_path = os.path.join(sam_ckpt_path, sam_files[0])
            print(f"Using SAM checkpoint from directory: {sam_ckpt_path}")
        else:
            raise FileNotFoundError(f"No SAM checkpoint found in directory: {sam_ckpt_path}")
    
    # Verify checkpoint file exists
    if not os.path.isfile(sam_ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint file not found: {sam_ckpt_path}")
    
    # Auto-detect model type from filename or path
    checkpoint_filename = os.path.basename(sam_ckpt_path).lower()
    checkpoint_fullpath = sam_ckpt_path.lower()
    
    # Check for fine-tuned SAM (default to vit_b)
    if 'sam_20260101_003941' in checkpoint_fullpath or 'best_model.pth' in checkpoint_filename:
        model_type = "vit_b"  # Fine-tuned SAM was trained with vit_b
        print(f"  Detected fine-tuned SAM checkpoint -> using model type: vit_b")
    elif 'vit_h' in checkpoint_filename or 'vit-h' in checkpoint_filename:
        model_type = "vit_h"
    elif 'vit_b' in checkpoint_filename or 'vit-b' in checkpoint_filename:
        model_type = "vit_b"
    elif 'vit_l' in checkpoint_filename or 'vit-l' in checkpoint_filename:
        model_type = "vit_l"
    else:
        raise ValueError(f"Cannot detect SAM model type from filename: {checkpoint_filename}. "
                        f"Expected filename to contain 'vit_h', 'vit_b', or 'vit_l'")
    
    print(f"Detected SAM model type: {model_type}")
    print(f"Loading SAM checkpoint: {sam_ckpt_path}")
    
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    # Auto-detect and load CLIP checkpoint if available
    clip_checkpoint = None
    
    # Priority 1: Fine-tuned SAM+CLIP from outputs
    finetuned_clip_paths = [
        "./outputs/sam_clip_20260101_012051/best_model.pth",  # Local path
        "/workspace/finetuning/outputs/sam_clip_20260101_012051/best_model.pth",  # Docker path
    ]
    
    for finetuned_clip_path in finetuned_clip_paths:
        if os.path.isfile(finetuned_clip_path):
            clip_checkpoint = finetuned_clip_path
            print(f"✓ Found fine-tuned SAM+CLIP checkpoint: {clip_checkpoint}")
            print(f"  Training: outputs/sam_clip_20260101_012051")
            break
    
    # Priority 2: CLIP in SAM checkpoint directory
    if clip_checkpoint is None:
        clip_checkpoint_dir = os.path.dirname(sam_ckpt_path) if sam_ckpt_path else "checkpoints"
        if os.path.isdir(clip_checkpoint_dir):
            clip_files = [f for f in os.listdir(clip_checkpoint_dir)
                         if 'clip' in f.lower() and f.endswith('.pth')]
            if clip_files:
                clip_files.sort()
                clip_checkpoint = os.path.join(clip_checkpoint_dir, clip_files[0])
                print(f"Auto-detected CLIP checkpoint: {clip_checkpoint}")
                print(f"  ⚠️  Warning: Using non-finetuned CLIP. For best results, use fine-tuned model.")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA-enabled GPU.")
    
    print("Initializing OpenCLIP model...")
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig(), checkpoint_path=clip_checkpoint)
    
    print(f"Loading SAM model ({model_type})...")
    
    # Check if this is a fine-tuned SAM checkpoint
    is_finetuned_sam = 'sam_20260101_003941' in sam_ckpt_path
    
    if is_finetuned_sam:
        print("  Loading fine-tuned SAM checkpoint...")
        checkpoint = torch.load(sam_ckpt_path, map_location='cpu')
        
        # Extract SAM state dict from fine-tuning format
        if 'model_state_dict' in checkpoint:
            sam_state_dict = checkpoint['model_state_dict']
            print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Validation IoU: {checkpoint.get('val_iou', 'unknown')}")
        else:
            sam_state_dict = checkpoint
        
        # Initialize SAM model without pretrained weights
        sam = sam_model_registry[model_type](checkpoint=None)
        sam.load_state_dict(sam_state_dict, strict=True)
        print("✓ Fine-tuned SAM weights loaded successfully")
    else:
        # Load pretrained SAM
        sam = load_sam_checkpoint(sam_ckpt_path, model_type)
        print("✓ Pretrained SAM model loaded successfully")
    
    sam = sam.to('cuda')
    print("✓ SAM model moved to CUDA")
    
    # Improved SAM Configuration for Fine-Grained Part Detection
    # Optimized for small car parts: emblems, handles, wheel spokes, mirror details, etc.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=96,  # ↑ 64 -> 96: Denser grid sampling for small parts
                             # Higher density helps detect wheel spokes, window frames, emblems
        pred_iou_thresh=0.6,  # Keep at 0.6: Allow more candidate masks
        box_nms_thresh=0.7,   # Keep at 0.7: Standard NMS threshold
        stability_score_thresh=0.75,  # ↓ 0.80 -> 0.75: More lenient for distant/small parts
        crop_n_layers=0,      # Disabled: Avoid custom SAM bug with crop_boxes
        crop_n_points_downscale_factor=1,
        min_mask_region_area=20,  # ↓ 50 -> 20: Detect very small parts (< 50 pixels)
                                  # Critical for distant cars or small details like logos
    )
    
    print(f"\n{'='*60}")
    print("SAM Mask Generator Configuration:")
    print(f"  - Sampling Grid: 96x96 (High density for small parts)")
    print(f"  - Min Mask Area: 20 pixels (Capture tiny details)")
    print(f"  - Stability Threshold: 0.75 (Balanced precision/recall)")
    print(f"{'='*60}\n")
    
    global dino_model
    dino_model = None

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    
    # CLIP-only mode for compatibility with CF3 (512 dimensions)
    use_dinov2 = False  # Disabled for CF3 compatibility
    use_mask_aggregation = True  # Enable Mask-CLIP Aggregation for better fine-grained recognition
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Configuration:")
    print(f"  - Model: OpenCLIP (ViT-B-32)")
    print(f"  - DINOv2 (Structural Features): Disabled")
    print(f"  - Mask-CLIP Aggregation (High-res Part Encoding): Enabled")
    print(f"  - Context-Aware Blur Cropping: Enabled (expand_ratio=0.10)")
    print(f"  - Feature Dimension: 512 (CLIP only)")
    print(f"{'='*60}\n")
    
    create(imgs, data_list, save_folder, use_dinov2=False, use_mask_aggregation=use_mask_aggregation)
