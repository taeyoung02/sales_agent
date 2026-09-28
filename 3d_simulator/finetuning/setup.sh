#!/bin/bash

# Setup script for SAM + CLIP fine-tuning environment

echo "=========================================="
echo "Setting up SAM + CLIP fine-tuning environment"
echo "=========================================="

# Check CUDA availability
echo ""
echo "Checking CUDA..."
nvcc --version
nvidia-smi

# Create virtual environment (optional)
# python3 -m venv venv
# source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 13.0 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install numpy opencv-python Pillow tqdm matplotlib tensorboard

# Install OpenCLIP
echo ""
echo "Installing OpenCLIP..."
pip install open-clip-torch

# Install Segment Anything
echo ""
echo "Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 << EOF
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import open_clip

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"PIL version: {Image.__version__}")
print(f"OpenCLIP available: {open_clip is not None}")

try:
    import segment_anything
    print(f"Segment Anything installed: ✓")
except ImportError:
    print(f"Segment Anything installed: ✗")
EOF

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python train.py --data-root ./VehicleSeg10K --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth"
echo ""
