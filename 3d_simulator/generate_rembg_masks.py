"""
Rembg-based Background Removal and Mask Generation

This script uses the rembg library with CUDA acceleration to generate high-quality masks.
Rembg supports multiple models for different use cases.

Usage:
    python generate_rembg_masks.py \
        --images_dir /workspace/data/my_project/images \
        --output_dir /workspace/data/my_project/masks \
        --model u2net \
        --device cuda

Available Models:
    - u2net (default): General purpose, good for most objects
    - u2netp: Lightweight version of u2net, faster
    - u2net_human_seg: Optimized for human segmentation
    - u2net_cloth_seg: Optimized for clothing
    - silueta: Optimized for general segmentation
    - isnet-general-use: High quality general purpose
    - isnet-anime: Optimized for anime characters
    - sam: Segment Anything Model (requires additional setup)

Requirements:
    pip install rembg[gpu] pillow numpy tqdm opencv-python
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


def setup_rembg(model_name="u2net", device="cuda"):
    """
    Setup rembg library with specified model
    
    Args:
        model_name: Name of the rembg model
        device: Device to use (cuda/cpu)
    
    Returns:
        Session object or None if setup fails
    """
    try:
        from rembg import remove, new_session
        import torch
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        
        print(f"Initializing rembg with model: {model_name}")
        print(f"Device: {device}")
        
        # Create session with specified model
        # For GPU acceleration, rembg will automatically use CUDA if available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = new_session(model_name, providers=providers)
        
        print(f"Rembg session initialized successfully")
        return session
        
    except ImportError:
        print("Error: rembg library not found.")
        print("Install with: pip install rembg[gpu]")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing rembg: {e}")
        sys.exit(1)


def generate_mask_rembg(image_path, session, output_type="mask", alpha_matting=False, 
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10):
    """
    Generate mask using rembg library
    
    Args:
        image_path: Path to input image
        session: Rembg session object
        output_type: Type of output - "mask" (binary), "alpha" (soft), or "rgba" (RGBA image)
        alpha_matting: Enable alpha matting for better edges
        alpha_matting_foreground_threshold: Foreground threshold for alpha matting
        alpha_matting_background_threshold: Background threshold for alpha matting
        alpha_matting_erode_size: Erosion size for alpha matting
    
    Returns:
        PIL Image - Generated mask (L mode for mask, RGBA for rgba output)
    """
    from rembg import remove
    
    # 1. Load image with PIL
    input_image = Image.open(image_path).convert("RGB")
    
    # 2. Remove background using rembg
    # rembg automatically returns RGBA image
    output_rgba = remove(
        input_image,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
        only_mask=False  # Get full RGBA output first
    )
    
    # 3. Extract alpha channel as mask
    if output_type == "rgba":
        # Return RGBA image with transparent background
        return output_rgba
    
    # Extract alpha channel
    alpha_channel = np.array(output_rgba)[:, :, 3]  # Get alpha channel
    
    if output_type == "alpha":
        # Return soft alpha mask (0-255 grayscale)
        mask_image = Image.fromarray(alpha_channel)
    else:  # output_type == "mask"
        # Return binary mask (0 or 255)
        # Threshold at 128 (middle value)
        binary_mask = (alpha_channel > 128).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask)
    
    return mask_image


def process_directory(images_dir, output_dir, session, output_type="mask",
                     alpha_matting=False, alpha_matting_params=None,
                     image_extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    """
    Process all images in a directory and generate masks
    
    Args:
        images_dir: Input directory containing images
        output_dir: Output directory for masks
        session: Rembg session
        output_type: Type of output - "mask", "alpha", or "rgba"
        alpha_matting: Enable alpha matting
        alpha_matting_params: Dictionary of alpha matting parameters
        image_extensions: Tuple of valid image extensions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {images_dir}")
        print(f"Looking for extensions: {image_extensions}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    print(f"Output type: {output_type}")
    print(f"Alpha matting: {alpha_matting}\n")
    
    # Default alpha matting parameters
    if alpha_matting_params is None:
        alpha_matting_params = {
            'alpha_matting_foreground_threshold': 240,
            'alpha_matting_background_threshold': 10,
            'alpha_matting_erode_size': 10
        }
    
    # Process each image
    success_count = 0
    failed_files = []
    
    for image_path in tqdm(image_files, desc="Generating rembg masks"):
        try:
            # Generate mask
            output_image = generate_mask_rembg(
                image_path,
                session,
                output_type=output_type,
                alpha_matting=alpha_matting,
                **alpha_matting_params
            )
            
            # Save output
            output_ext = '.png'  # Always use PNG for masks to preserve alpha
            output_path = os.path.join(output_dir, image_path.stem + output_ext)
            output_image.save(output_path)
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {image_path.name}: {str(e)}")
            failed_files.append(image_path.name)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Rembg Mask Generation Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")
    
    print(f"\nMasks saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks using rembg library with CUDA acceleration"
    )
    
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output masks"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="u2net",
        choices=[
            "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
            "silueta", "isnet-general-use", "isnet-anime", "sam"
        ],
        help="Rembg model to use (default: u2net)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    
    parser.add_argument(
        "--output_type",
        type=str,
        default="mask",
        choices=["mask", "alpha", "rgba"],
        help="Output type: mask (binary), alpha (soft), or rgba (RGBA image) (default: mask)"
    )
    
    parser.add_argument(
        "--alpha_matting",
        action="store_true",
        help="Enable alpha matting for better edge refinement"
    )
    
    parser.add_argument(
        "--alpha_matting_foreground_threshold",
        type=int,
        default=240,
        help="Foreground threshold for alpha matting (default: 240)"
    )
    
    parser.add_argument(
        "--alpha_matting_background_threshold",
        type=int,
        default=10,
        help="Background threshold for alpha matting (default: 10)"
    )
    
    parser.add_argument(
        "--alpha_matting_erode_size",
        type=int,
        default=10,
        help="Erosion size for alpha matting (default: 10)"
    )
    
    parser.add_argument(
        "--image_extensions",
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
        help="Image file extensions to process (default: .jpg .jpeg .png)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)
    
    # Setup rembg session
    session = setup_rembg(args.model, args.device)
    
    # Prepare alpha matting parameters
    alpha_matting_params = {
        'alpha_matting_foreground_threshold': args.alpha_matting_foreground_threshold,
        'alpha_matting_background_threshold': args.alpha_matting_background_threshold,
        'alpha_matting_erode_size': args.alpha_matting_erode_size
    }
    
    # Process all images
    process_directory(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        session=session,
        output_type=args.output_type,
        alpha_matting=args.alpha_matting,
        alpha_matting_params=alpha_matting_params,
        image_extensions=tuple(args.image_extensions)
    )


if __name__ == "__main__":
    main()
