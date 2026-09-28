"""
ESRGAN-based Image Preprocessing with Light Reflection & Bloom Removal
======================================================================

This script applies GAN-based preprocessing to remove:
1. Light reflections (using Lab color space + CLAHE)
2. Light bloom/glare (using ESRGAN generator)

Usage:
    python esrgan_preprocessing.py \
        --images_dir /workspace/data/my_project/input \
        --output_dir /workspace/data/my_project/enhanced \
        --weights /workspace/weights/esrgan_model.pth \
        --scale_factor 2 \
        --remove_reflection \
        --remove_bloom \
        --device cuda

Requirements:
    pip install torch torchvision opencv-python pillow numpy tqdm
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functools


# ============================================================================
# Model Architecture (from notebook)
# ============================================================================

def initialize_weights(net_l, scale=1):
    """Initialize network weights with Kaiming normal distribution"""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    """Create sequential layers from a block"""
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block with 5 convolutional layers"""
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Generator(nn.Module):
    """ESRGAN Generator for super-resolution and deblurring"""
    def __init__(self, scale_factor, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(Generator, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = scale_factor

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


# ============================================================================
# Light Reflection Removal (from notebook)
# ============================================================================

def remove_light_reflection(image):
    """
    Remove light reflections using Lab color space + CLAHE
    
    Args:
        image: PIL Image or numpy array (RGB)
    
    Returns:
        numpy array: Image with reflections removed (RGB)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert RGB to BGR (OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert BGR to Lab color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Split L, a, b channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    # This suppresses bright areas (reflections) and equalizes lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Additional suppression of bright areas
    # Reduce brightness above threshold
    threshold = 220
    mask = l_channel > threshold
    cl[mask] = np.clip(cl[mask] * 0.7, 0, 255).astype(np.uint8)
    
    # Merge channels
    enhanced_lab = cv2.merge([cl, a_channel, b_channel])
    
    # Convert Lab to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert BGR to RGB for output
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    return enhanced_rgb


# ============================================================================
# Image Processing Pipeline
# ============================================================================

def get_image_files(directory):
    """Get all image files from directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', 
                 '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
    
    return sorted(image_files)


def resize_image_if_needed(img, max_width=3240):
    """
    Resize image if width exceeds max_width, maintaining aspect ratio
    
    Args:
        img: PIL Image
        max_width: Maximum width in pixels
    
    Returns:
        PIL Image: Resized image or original if no resize needed
    """
    width, height = img.size
    
    if width > max_width:
        # Calculate new height to maintain aspect ratio
        ratio = max_width / width
        new_height = int(height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
        print(f"  Resized from {width}x{height} to {max_width}x{new_height}")
    
    return img


def process_image(image_path, model, device, remove_reflection=True, remove_bloom=True, max_width=None):
    """
    Process a single image with reflection and/or bloom removal
    
    Args:
        image_path: Path to input image
        model: ESRGAN generator model
        device: torch device (cuda/cpu)
        remove_reflection: Whether to remove reflections
        remove_bloom: Whether to remove bloom (requires model)
        max_width: Maximum width for resizing (None to disable)
    
    Returns:
        PIL Image: Processed image
    """
    # Load image and handle EXIF orientation
    img = Image.open(image_path)
    
    # Apply EXIF orientation if present (fixes rotation issues)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # If EXIF processing fails, continue with original
    
    img = img.convert('RGB')
    
    # Step 0: Resize if needed (before processing to save memory)
    if max_width is not None:
        img = resize_image_if_needed(img, max_width)
    
    # Step 1: Remove reflections (if enabled)
    if remove_reflection:
        img_array = remove_light_reflection(img)
        img = Image.fromarray(img_array)
    
    # Step 2: Remove bloom using ESRGAN (if enabled and model available)
    if remove_bloom and model is not None:
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Run through model
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert back to PIL
        output = output.squeeze(0).cpu().clamp(0, 1)
        output_array = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(output_array)
    
    return img


def load_model(weights_path, scale_factor, device):
    """
    Load ESRGAN model from weights file
    
    Args:
        weights_path: Path to model weights (.pth file)
        scale_factor: Upscaling factor (2 or 4)
        device: torch device
    
    Returns:
        model: Loaded model in eval mode, or None if weights not found
    """
    if weights_path is None or not os.path.exists(weights_path):
        print(f"⚠️  Model weights not found at: {weights_path}")
        print("⚠️  Bloom removal will be skipped (only reflection removal will apply)")
        return None
    
    print(f"Loading model from: {weights_path}")
    model = Generator(scale_factor=scale_factor, in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    return model


def main():
    parser = argparse.ArgumentParser(description='ESRGAN-based Image Preprocessing')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed images')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to ESRGAN model weights (.pth file)')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 4],
                       help='Upscaling factor (2 or 4)')
    parser.add_argument('--remove_reflection', action='store_true',
                       help='Enable light reflection removal')
    parser.add_argument('--remove_bloom', action='store_true',
                       help='Enable light bloom removal (requires weights)')
    parser.add_argument('--max_width', type=int, default=None,
                       help='Maximum width in pixels (e.g., 1080). Images wider than this will be resized')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 60)
    print("ESRGAN Image Preprocessing")
    print("=" * 60)
    print(f"Input directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reflection removal: {args.remove_reflection}")
    print(f"Bloom removal: {args.remove_bloom}")
    print(f"Max width: {args.max_width if args.max_width else 'None (no resizing)'}")
    print(f"Device: {args.device}")
    print()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model (if bloom removal enabled)
    model = None
    if args.remove_bloom:
        model = load_model(args.weights, args.scale_factor, device)
        if model is None and not args.remove_reflection:
            print("ERROR: No processing method available (both model and reflection removal disabled)")
            sys.exit(1)
    
    # Get image files
    image_files = get_image_files(args.images_dir)
    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    print("Processing images...")
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Process image
            output_img = process_image(
                img_path, 
                model, 
                device, 
                remove_reflection=args.remove_reflection,
                remove_bloom=args.remove_bloom,
                max_width=args.max_width
            )
            
            # Save result
            output_path = os.path.join(args.output_dir, img_path.name)
            output_img.save(output_path, quality=95)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            continue
    
    print()
    print("=" * 60)
    print("✓ Processing complete!")
    print(f"  Output saved to: {args.output_dir}")
    print(f"  Processed {len(image_files)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
