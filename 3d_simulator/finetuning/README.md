# SAM + CLIP Fine-tuning for VehicleSeg10K

Vehicle part segmentation using SAM (Segment Anything Model) and CLIP fine-tuning on VehicleSeg10K dataset.

## 🐳 Quick Start with Docker (Recommended)

```bash
# 1. Build Docker image
./docker-build.sh

# 2. Test environment
./docker-test.sh

# 3. Start training
./docker-train.sh

# 4. Monitor with TensorBoard
./docker-tensorboard.sh
```

**👉 For detailed Docker instructions, see [DOCKER_GUIDE.md](DOCKER_GUIDE.md)**

---

## Features

- **SAM for Mask Generation**: Automatically generates high-quality segmentation masks
- **CLIP for Classification**: Fine-tuned CLIP model classifies each mask into vehicle parts
- **Background-Aware Training**: "Others" class uses averaged embeddings from background-related words
  - Instead of generic "Others", uses semantic mean of 12 background words (asphalt, road, surroundings, etc.)
  - Leverages CLIP's pre-trained text-image alignment for better background separation
  - See [BACKGROUND_EMBEDDING.md](BACKGROUND_EMBEDDING.md) for details
- **Context-Aware Blur Cropping**: Advanced preprocessing for better feature extraction
  - Expands bbox by 30% to preserve contextual information
  - Applies Gaussian blur to background while keeping mask region sharp
  - Improves fine-grained recognition of small parts (emblems, handles, spokes)
  - See [CONTEXT_AWARE_BLUR.md](CONTEXT_AWARE_BLUR.md) for details
- **GPU Accelerated**: Full CUDA support for fast training and inference
- **13 Vehicle Part Classes**:
  - Back window
  - Foreground
  - Front window
  - Left back door
  - Left back window
  - Left front door
  - Left front window
  - Plate
  - Right back door
  - Right back window
  - Right front door
  - Right front window
  - Wheel
  - Others (for unmatched regions)

## Dataset Structure

```
VehicleSeg10K/
├── images/
│   ├── train/
│   └── valid/
└── annotations/
    ├── train/
    └── valid/
```

## Installation

### Option 1: Docker (Recommended - No Base Environment Pollution)
See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete Docker setup.

### Option 2: Direct Installation
```bash
pip install -r requirements.txt
```

## Training

### With Docker
```bash
./docker-train.sh
```

### Without Docker

Basic training:
```bash
python train.py \
    --data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --device cuda \
    --gpu-id 0
```

Advanced options:
```bash
python train.py \
    --data-root ./VehicleSeg10K \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --sam-model-type vit_b \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --batch-size 4 \
    --num-workers 4 \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --save-dir ./outputs \
    --save-every 5 \
    --device cuda \
    --gpu-id 0
```

## Inference

Run inference on a single image:
```bash
python inference.py \
    --checkpoint ./outputs/sam_clip_XXXXXX/best_model.pth \
    --image ./test_image.jpg \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --output-dir ./outputs/inference \
    --device cuda \
    --gpu-id 0 \
    --show
```

## Model Architecture

1. **SAM (Frozen)**: Generates high-quality segmentation masks
2. **CLIP Vision Encoder (Trainable)**: Extracts features from masked regions
3. **Classification Head (Trainable)**: Classifies masks into vehicle parts

## Training Strategy

- **SAM**: Frozen, used only for mask generation
- **CLIP Text Encoder**: Frozen
- **CLIP Vision Encoder**: Fine-tuned on vehicle part images
- **Classification Head**: Trained from scratch

## GPU Acceleration

The code fully supports CUDA acceleration:
- Automatic GPU detection
- Mixed precision training ready
- Multi-GPU support (specify with --gpu-id)
- Pinned memory for faster data loading

## Monitoring

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir ./outputs/sam_clip_XXXXXX/logs
```

## Debugging & Visualization

After training, use the debugging tool to visualize SAM masks and CLIP predictions:

### Quick Debug (Random 5 samples from validation set)
```bash
./docker-debug.sh
```

### Debug Specific Image
```bash
# Inside Docker container
./docker-run.sh

# Then inside container:
python debugging.py \
    --image ./VehicleSeg10K/images/valid/000044576.jpg \
    --checkpoint ./outputs/best_model.pth \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --data-root ./VehicleSeg10K \
    --output-dir debug_output
```

### Visualization Outputs

For each image, the debugging tool generates 4 visualizations:

1. **`*_1_sam_masks.png`**: SAM mask generation analysis
   - All generated masks overlaid on image
   - Mask boundaries
   - Stability scores heatmap
   - Predicted IoU heatmap
   - Mask size distribution

2. **`*_2_context_aware_cropping.png`**: Context-aware blur cropping comparison
   - Original image with bboxes (tight vs expanded)
   - Tight crop with black background (original method)
   - Expanded crop without blur
   - Expanded crop WITH blur (final method)
   - Difference heatmap

3. **`*_3_clip_predictions.png`**: CLIP classification results
   - Predictions overlaid with labels and confidence scores
   - Confidence distribution per mask
   - Label distribution histogram

4. **`*_4_gt_comparison.png`**: Ground truth comparison (if annotations available)
   - Ground truth masks
   - SAM + CLIP predictions
   - Color-coded matching (Green=Correct, Red=Wrong, Gray=No GT)
   - Confusion matrix

### Example Debug Output

```bash
# After running docker-debug.sh
ls debug_output/

# Output:
# 000044576_1_sam_masks.png
# 000044576_2_context_aware_cropping.png
# 000044576_3_clip_predictions.png
# 000044576_4_gt_comparison.png
# 000044634_1_sam_masks.png
# ...
```

## Output Structure

```
outputs/
└── sam_clip_YYYYMMDD_HHMMSS/
    ├── logs/                      # TensorBoard logs
    ├── checkpoint_epoch_5.pth     # Periodic checkpoints
    ├── checkpoint_epoch_10.pth
    ├── ...
    └── best_model.pth            # Best model by validation accuracy
```

## Performance

Training on CUDA-enabled GPU:
- Expected training time: ~2-3 hours for 50 epochs (depending on GPU)
- Validation accuracy: Typically >85% for vehicle part classification

## Citation

If you use this code, please cite:
- SAM: [Segment Anything](https://github.com/facebookresearch/segment-anything)
- CLIP: [OpenCLIP](https://github.com/mlfoundations/open_clip)
- VehicleSeg10K dataset
