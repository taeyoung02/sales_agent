#
# Lazy camera loading utility to prevent OOM during scene initialization
#

from scene.cameras import Camera
from utils.camera_utils import loadCam
from PIL import Image
import torch
import os
import numpy as np

class LazyCameraList:
    """
    Lazy loading wrapper for camera list.
    Only loads cameras when accessed, preventing OOM during scene initialization.
    """
    def __init__(self, cam_infos, resolution_scale, args):
        self.cam_infos = cam_infos
        self.resolution_scale = resolution_scale
        self.args = args
        self._cache = {}
        
        # Detect mask directory
        self.masks_folder = None
        if hasattr(args, 'source_path') and args.source_path:
            source_path = args.source_path
            # Try to find masks folder
            # Case 1: source_path is *_langsplat, look in parent/project_name/masks
            if source_path.endswith('_langsplat'):
                dataset_parent = os.path.dirname(source_path.rstrip('/'))
                dataset_name = os.path.basename(source_path.rstrip('/'))
                project_name = dataset_name[:-len('_langsplat')]
                potential_masks = os.path.join(dataset_parent, project_name, 'masks')
            else:
                # Case 2: source_path/masks
                potential_masks = os.path.join(source_path, 'masks')
            
            if os.path.isdir(potential_masks):
                self.masks_folder = potential_masks
                print(f"✓ Binary masks detected: {self.masks_folder}")
            else:
                print(f"⚠️  No binary masks found, proceeding without masking")
    
    def __len__(self):
        return len(self.cam_infos)
    
    def __getitem__(self, idx):
        """Load camera on-demand WITHOUT caching to prevent memory accumulation"""
        if isinstance(idx, slice):
            # Handle slicing
            indices = range(*idx.indices(len(self.cam_infos)))
            return [self[i] for i in indices]
        
        # Single index access - NO CACHING
        cam_info = self.cam_infos[idx]
        
        # Lazy load image and semantic feature if not already loaded
        if cam_info.image is None:
            # Load image on-demand
            try:
                image = Image.open(cam_info.image_path)
                
                # Apply binary mask if available (matching LangSplat preprocessing)
                mask_applied = False
                if self.masks_folder:
                    # Get mask filename (replace extension with .png)
                    mask_filename = os.path.splitext(os.path.basename(cam_info.image_path))[0] + '.png'
                    mask_path = os.path.join(self.masks_folder, mask_filename)
                    
                    if os.path.isfile(mask_path):
                        # Load mask
                        mask_img = Image.open(mask_path).convert('L')
                        
                        # Resize mask to match image size
                        if mask_img.size != image.size:
                            mask_img = mask_img.resize(image.size, Image.NEAREST)
                        
                        # Convert to numpy for processing
                        image_np = np.array(image)
                        mask_np = np.array(mask_img)
                        
                        # Threshold to binary (0 or 255)
                        mask_binary = (mask_np > 127).astype(np.uint8) * 255
                        
                        # Apply mask (set background to black)
                        if image_np.ndim == 3:  # RGB image
                            mask_3ch = np.stack([mask_binary] * 3, axis=-1)
                            image_np = (image_np * (mask_3ch / 255.0)).astype(np.uint8)
                        else:  # Grayscale image
                            image_np = (image_np * (mask_binary / 255.0)).astype(np.uint8)
                        
                        # Convert back to PIL
                        image = Image.fromarray(image_np)
                        mask_applied = True
                        
                        # Debug: Log masking for first camera only
                        if idx == 0:
                            print(f"  ✓ Runtime masking applied: {mask_filename}")
                    else:
                        if idx == 0:
                            print(f"  ⚠️  Mask file not found: {mask_path}")
                
                # IMPORTANT: If no runtime masking but LangSplat preprocessing was done,
                # the images should already be masked in the images folder
                if not mask_applied and idx == 0:
                    print(f"  ℹ️  No runtime masking (image should be pre-masked by LangSplat): {os.path.basename(cam_info.image_path)}")
                        
            except FileNotFoundError:
                print(f"⚠️  Warning: Image file not found: {cam_info.image_path}, skipping camera {idx}")
                return None
            
            # Load semantic feature on-demand
            try:
                semantic_feature = torch.load(cam_info.semantic_feature_path, weights_only=True)
            except FileNotFoundError:
                print(f"⚠️  Warning: Semantic feature file not found: {cam_info.semantic_feature_path}, skipping camera {idx}")
                return None
            
            # Create a new CameraInfo with loaded data
            from scene.dataset_readers import CameraInfo
            cam_info = CameraInfo(
                uid=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FovY=cam_info.FovY,
                FovX=cam_info.FovX,
                image=image,
                image_path=cam_info.image_path,
                image_name=cam_info.image_name,
                width=cam_info.width,
                height=cam_info.height,
                semantic_feature=semantic_feature,
                semantic_feature_path=cam_info.semantic_feature_path,
                semantic_feature_name=cam_info.semantic_feature_name
            )
        
        # Create Camera object (NO CACHING - always create new)
        return loadCam(self.args, idx, cam_info, self.resolution_scale)
    
    def __iter__(self):
        """Support iteration"""
        for i in range(len(self)):
            yield self[i]
    
    def pop(self, index=-1):
        """Support pop operation for compatibility with viewpoint_stack.pop()"""
        # Get the camera at index
        camera = self[index]
        
        # Remove from cam_infos
        if index == -1 or index == len(self.cam_infos) - 1:
            self.cam_infos = self.cam_infos[:-1]
        else:
            self.cam_infos = self.cam_infos[:index] + self.cam_infos[index+1:]
        
        # Remove from cache if present
        if index in self._cache:
            del self._cache[index]
        
        # Rebuild cache with new indices
        new_cache = {}
        for old_idx, cam in self._cache.items():
            if old_idx < index:
                new_cache[old_idx] = cam
            else:
                new_cache[old_idx - 1] = cam
        self._cache = new_cache
        
        return camera
    
    def clear_cache(self):
        """Clear cached cameras to free memory"""
        self._cache.clear()
    
    def copy(self):
        """Return a shallow copy that shares the same cam_infos but has separate cache"""
        new_list = LazyCameraList(self.cam_infos, self.resolution_scale, self.args)
        return new_list
