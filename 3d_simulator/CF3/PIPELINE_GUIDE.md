# 3D Reconstruction Pipeline Guide

Complete guide for running the full 3D reconstruction pipeline:  
**COLMAP → 3D Gaussian Splatting → LangSplat → CF3 (Compact Feature Field)**

---

## Pipeline Overview

```
Input Images → COLMAP → 3DGS → LangSplat → CF3 → 3D Semantic Scene
```

### Phase 1: COLMAP + 3D Gaussian Splatting
- **Input**: Raw images
- **Output**: 3D Gaussian representation + camera poses
- **Duration**: ~10 minutes (COLMAP CPU) + ~7 minutes (3DGS training)

### Phase 2: LangSplat Preprocessing
- **Input**: Images + COLMAP data + 3DGS output
- **Output**: Semantic features (.npy files)
- **Duration**: ~5-15 minutes (depends on SAM model: ViT-B is faster)

### Phase 3: CF3 (Compact Feature Field)
- **Input**: LangSplat features + 3DGS output
- **Output**: Compressed semantic 3D scene
- **Duration**: ~10-20 minutes (feature conversion + autoencoder training + CF3 training)

---

## Prerequisites

### Docker Image
```bash
# Build the pipeline Docker image
cd /home/kolon/Desktop/work/Pipeline
docker build -t 3d-reconstruction-pipeline:latest -f Dockerfile .
```

### Required Files
- **Images**: Place in `data/<project_name>/input/` or `data/<project_name>/images/`
- **SAM Checkpoint**: Download to `data/checkpoints/`

### Download SAM Model (Optional)
```bash
cd data/checkpoints

# Recommended: ViT-B (358 MB, fast)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Alternative: ViT-H (2.4 GB, highest quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Alternative: ViT-L (1.2 GB, balanced)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

---

## Directory Structure

### Input Structure
```
data/
├── checkpoints/
│   ├── sam_vit_b_01ec64.pth          # SAM model (auto-detected)
│   └── sam_vit_h_4b8939.pth          # Alternative SAM model
└── <project_name>/
    └── input/                         # Or images/
        ├── image_001.jpg
        ├── image_002.jpg
        └── ...
```

### Output Structure (After Full Pipeline)
```
data/
├── <project_name>/                    # Phase 1 output
│   ├── images/                        # Processed images
│   ├── sparse/                        # COLMAP sparse reconstruction
│   │   └── 0/                         # Auto-created subdirectory
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   ├── distorted/                     # Intermediate COLMAP data
│   └── output/                        # 3DGS training output
│       ├── cameras.json
│       ├── cfg_args
│       ├── chkpnt30000.pth
│       ├── input.ply
│       └── point_cloud/
│           └── iteration_30000/
│               └── point_cloud.ply
│           └── iteration_7000/
│               └── point_cloud.ply
│
├── <project_name>_langsplat/         # Phase 2 output
│   ├── images/                        # Flattened image copies
│   ├── sparse/                        # COLMAP data copy
│   ├── output/                        # 3DGS output copy
│   ├── language_features/             # LangSplat output (.npy)
│   │   ├── frame_00000_f.npy
│   │   ├── frame_00000_s.npy
│   │   └── ...
│   └── langsplat_features/            # CF3 input (.pt)
│       ├── frame_00000_fmap_CxHxW.pt
│       └── ...
│
└── <project_name>_cf3/                # Phase 3 output
    ├── train_before_merging/
    ├── train_after_merging/
    └── point_cloud/
        └── iteration_*/
            ├── point_cloud.ply
            └── feature_field.ply
```

---

## Phase 1: COLMAP + 3D Gaussian Splatting

### Step 1: Prepare Images
```bash
# Create project directory
PROJECT_NAME="my_project"
mkdir -p data/${PROJECT_NAME}/input

# Copy your images
cp /path/to/your/images/*.jpg data/${PROJECT_NAME}/input/

# Verify image count
ls data/${PROJECT_NAME}/input/ | wc -l
```

### Step 2: Run COLMAP Pipeline

```bash
docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=64gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/data:/workspace/data \
    -e PROJECT_NAME=${PROJECT_NAME} \
    3d-reconstruction-pipeline:latest \
    bash -c '
cd /workspace/data/${PROJECT_NAME}

rm -f database.db
rm -rf sparse distorted

echo ""
echo "=========================================="
echo "Step 1/4: COLMAP Feature Extraction (CPU)"
echo "=========================================="
colmap feature_extractor \
    --database_path database.db \
    --image_path input \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.num_threads 8 \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192 \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo ""
echo "=========================================="
echo "Step 2/4: COLMAP Feature Matching (CPU)"
echo "=========================================="
colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 0 \
    --SiftMatching.num_threads 8

echo ""
echo "=========================================="
echo "Step 3/4: COLMAP Sparse Reconstruction"
echo "=========================================="
mkdir -p sparse
colmap mapper \
    --database_path database.db \
    --image_path input \
    --output_path sparse

if [ ! -d sparse/0 ]; then
    mkdir -p sparse/0
    mv sparse/*.bin sparse/0/ 2>/dev/null || true
    mv sparse/*.txt sparse/0/ 2>/dev/null || true
fi

if [ ! -f sparse/0/cameras.bin ]; then
    echo "ERROR: Reconstruction failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 4/4: COLMAP Image Undistortion"
echo "=========================================="
colmap image_undistorter \
    --image_path input \
    --input_path sparse/0 \
    --output_path dense \
    --output_type COLMAP

if [ ! -d dense/sparse ]; then
    echo "ERROR: Undistortion failed"
    exit 1
fi

# Organize as expected by 3DGS
mkdir -p images
cp -r dense/images/* images/ 2>/dev/null || cp input/* images/
mkdir -p sparse/0
cp dense/sparse/*.bin sparse/0/
cp dense/sparse/*.txt sparse/0/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "COLMAP Pipeline Completed!"
echo "=========================================="
'
```

**COLMAP Parameters:**
- `--SiftExtraction.use_gpu 0`: Force CPU mode (required for headless servers)
- `--SiftMatching.use_gpu 0`: Force CPU matching
- `--ImageReader.single_camera 1`: Assume single camera model
- Expected output: `sparse/0/` directory with `cameras.bin`, `images.bin`, `points3D.bin`

### Step 3: Train 3D Gaussian Splatting
```bash
docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=64gb \
    -v $(pwd)/data:/workspace/data \
    -e PROJECT_NAME=${PROJECT_NAME} \
    -e ITERATIONS=${ITERATIONS:-30000} \
    3d-reconstruction-pipeline:latest \
    bash -c "
cd /workspace/gaussian-splatting

python convert.py -s /workspace/data/\${PROJECT_NAME} --skip_matching

python train.py \
    -s /workspace/data/\${PROJECT_NAME} \
    -m /workspace/data/\${PROJECT_NAME}/output \
    --iterations \${ITERATIONS} \
    --checkpoint_iterations \${ITERATIONS}
"
```

**3DGS Training Parameters:**
- `-s`: Source path (COLMAP data directory)
- `-m`: Model output path
- `--iterations 7000`: Total training iterations (default: 30000)
  - **7000**: Quick training (~7 min)
  - **15000**: Medium quality (~15 min)
  - **30000**: High quality (~30 min)
- `--checkpoint_iterations`: When to save checkpoints
- `--test_iterations`: When to run test evaluation
- `--save_iterations`: Which iterations to save

**Performance:**
- Speed: ~14-15 it/s on GB10 GPU
- Memory: ~4-6 GB VRAM
- Duration: ~7 minutes for 7000 iterations

### Verify Phase 1 Output
```bash
ls -lh data/${PROJECT_NAME}/output/
# Should see: cameras.json, cfg_args, chkpnt7000.pth, point_cloud/

ls -lh data/${PROJECT_NAME}/output/point_cloud/iteration_30000/
# Should see: point_cloud.ply
```

---

## Phase 2: LangSplat Preprocessing

### Memory-Optimized Command (Recommended)
```bash
docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    --shm-size=64gb \
    --memory=96g \
    --memory-swap=96g \
    -v $(pwd)/data:/workspace/data \
    -e PROJECT_NAME=${PROJECT_NAME} \
    3d-reconstruction-pipeline:latest \
    bash -c "
LANGSPLAT_DIR=\"/workspace/data/\${PROJECT_NAME}_langsplat\"

echo 'Step 1: Creating directory structure'
rm -rf \${LANGSPLAT_DIR}
mkdir -p \${LANGSPLAT_DIR}/images

echo 'Step 2: Copying images'
find /workspace/data/\${PROJECT_NAME}/images -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPG' -o -name '*.PNG' \) -exec cp {} \${LANGSPLAT_DIR}/images/ \;

echo 'Step 3: Verifying copied images'
IMAGE_COUNT=\$(find \${LANGSPLAT_DIR}/images -type f | wc -l)
echo \"Found \${IMAGE_COUNT} images in \${LANGSPLAT_DIR}/images/\"

if [ \${IMAGE_COUNT} -eq 0 ]; then
    echo 'ERROR: No images copied!'
    echo 'Source directory contents:'
    ls -la /workspace/data/\${PROJECT_NAME}/images/
    exit 1
fi

echo 'Step 4: Copying COLMAP data'
cp -r /workspace/data/\${PROJECT_NAME}/sparse \${LANGSPLAT_DIR}/

echo 'Step 5: Running LangSplat preprocessing'
cd /workspace/LangSplat

python preprocess.py --dataset_path \${LANGSPLAT_DIR} --resolution 2048 --batch_size 4
"
```

**Memory Management:**
- `--memory=96g`: Limits container to 96GB RAM (adjust based on your system)
- `--memory-swap=96g`: Prevents swap usage (keeps same as memory)
- `--shm-size=64gb`: Shared memory for IPC
- `--batch_size`: Number of images processed simultaneously (default: 4)
  - Lower batch size = less memory usage, slower processing
  - Recommended: 2-4 for high-res images, 4-8 for smaller images

**LangSplat Parameters:**
- `--dataset_path`: Path to LangSplat data directory
- `--batch_size`: Images to process at once (default: 4)
  - Use `--batch_size 2` or `1` for very large images or limited RAM
- `--no_resize`: Keep original image size (high memory usage)
- `--resolution <pixels>`: Max dimension in pixels (default: -1 for auto 1080P)
- `--sam_ckpt_path`: (Optional) Path to SAM checkpoint
  - If not specified, auto-detects from `/workspace/data/checkpoints/`
- `--sam_checkpoint_dir`: (Optional) Custom checkpoint directory

**Resolution Options:**
```bash
# Default: Auto-resize >1080P to 1080P
python preprocess.py --dataset_path ${LANGSPLAT_DIR} --batch_size 4

# Keep original resolution (⚠️ High memory!)
python preprocess.py --dataset_path ${LANGSPLAT_DIR} --no_resize --batch_size 2

# Custom resolution (2048px max dimension)
python preprocess.py --dataset_path ${LANGSPLAT_DIR} --resolution 2048 --batch_size 4
```

**SAM Model Auto-Detection:**
- Searches `/workspace/data/checkpoints/` by default
- Detects model type (vit_h/vit_b/vit_l) from filename
- Uses first checkpoint alphabetically if multiple exist
- Example: `sam_vit_b_01ec64.pth` → detects as `vit_b`

**Performance & Memory:**
| Configuration | Time/Image | RAM Usage | VRAM Usage |
|--------------|------------|-----------|------------|
| ViT-B + 1080P + batch=4 | 5-10s | ~8-12GB | ~6-8GB |
| ViT-B + Original + batch=2 | 15-20s | ~20-30GB | ~8-10GB |
| ViT-H + 1080P + batch=4 | 15-20s | ~12-16GB | ~8-10GB |
| ViT-H + Original + batch=2 | 30-40s | ~30-40GB | ~10-12GB |

**Troubleshooting:**
- **Out of Memory**: Reduce `--batch_size` to 2 or 1
- **Still OOM**: Add `--resolution 1080` to force downscaling
- **Timeout errors**: Container using all RAM → Add `--memory` limit
- **Slow processing**: Use ViT-B instead of ViT-H

### Verify Phase 2 Output
```bash
ls -la data/${PROJECT_NAME}_langsplat/language_features/
# Should see: frame_*_f.npy and frame_*_s.npy files (2 files per image)

# Count feature files
ls data/${PROJECT_NAME}_langsplat/language_features/*.npy | wc -l
# Should be: (number of images) × 2
```

---

## Phase 3: CF3 (Compact Feature Field)

### Step 1: Convert LangSplat Features to CF3 Format
```bash
docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=64gb \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/CF3:/workspace/CF3 \
    3d-reconstruction-pipeline:latest \
    bash -c "
cd /workspace/CF3
mkdir -p /workspace/data/${PROJECT_NAME}_langsplat/langsplat_features
python langsplat_feature_convert.py \
    /workspace/data/${PROJECT_NAME}_langsplat/language_features \
    /workspace/data/${PROJECT_NAME}_langsplat/langsplat_features \
    3
"
```

**Feature Conversion Parameters:**
- Argument 1: Input directory (language_features with .npy files)
- Argument 2: Output directory (langsplat_features for .pt files)
- Argument 3: SAM level (0, 1, 2, or 3)
  - **0**: Coarsest level (recommended, fastest)
  - **1-3**: Finer levels (slower, more detail)

**Output:**
- Converts `frame_*_f.npy` + `frame_*_s.npy` → `frame_*_fmap_CxHxW.pt`
- One `.pt` file per frame

### Verify Feature Conversion
```bash
ls -la data/${PROJECT_NAME}_langsplat/langsplat_features/
# Should see: frame_*_fmap_CxHxW.pt files (1 file per image)

# Count converted features
ls data/${PROJECT_NAME}_langsplat/langsplat_features/*.pt | wc -l
# Should equal number of images
```

### Step 2: Run CF3 Training

#### Standard Command (Recommended)
```bash
docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/CF3:/workspace/CF3 \
    3d-reconstruction-pipeline:latest \
    bash -c "
cd /workspace/CF3
mkdir -p /workspace/data/${PROJECT_NAME}_cf3
python compact_feature_field.py \
    -s /workspace/data/${PROJECT_NAME}_langsplat \
    -m /workspace/data/${PROJECT_NAME}/output \
    -f langsplat \
    -o /workspace/data/${PROJECT_NAME}_cf3 \
    --antialiasing \
    --finetune_decoder \
    --normalize_feature \
    --iterations 30000
"

**Key Improvement: Lazy Camera Loading**
- **Problem Solved**: Previous version loaded ALL cameras into RAM at once, causing OOM
- **Solution**: Cameras are now loaded on-demand in batches during processing
- **Memory Savings**: Only processes batch_size cameras at a time, then releases memory
- **Automatic Batching**: Batch size is automatically adjusted based on available RAM
- **No Manual Config**: No need for `--max_images`, `--skip_every_n`, or `--image_downsample_factor`
- **Impact**: Can process 100+ high-res images on 64GB RAM systems
```

**Performance**: ~20-30 minutes for full training

#### Common Parameter Adjustments

**For Limited Memory (32-64GB RAM):**
```bash
# Add these parameters to the standard command above
--max_images 100 \
--image_downsample_factor 2.0 \
--compress_batch_size 32
```

**For Quick Testing:**
```bash
# Replace --iterations and add these parameters
--iterations 3000 \
--max_images 20 \
--skip_every_n 2 \
--compress_epoch 10 \
--skip_test
```

**For Maximum Quality:**
```bash
# Add these parameters to the standard command
--use_render_feature \
--filter_var \
--compress_epoch 50 \
--similarity_threshold 0.95
```

**For Aggressive Compression (Smaller Output):**
```bash
# Add these parameters to the standard command
--contrib_threshold 0.0005 \
--similarity_threshold 0.85 \
--merge_interval 50
```

---

## CF3 Parameters Reference

### Memory & Image Loading Options Summary

| Use Case | RAM | Command Options |
|----------|-----|-----------------|
| **Full Quality** | 128GB | `--image_downsample_factor 1.0` |
| **High Quality** | 96GB | `--image_downsample_factor 1.0 --skip_every_n 2` |
| **Balanced** | 64GB | `--image_downsample_factor 2.0 --max_images 100` |
| **Low Memory** | 32GB | `--image_downsample_factor 4.0 --max_images 50` |
| **Quick Test** | 16GB | `--max_images 20 --image_downsample_factor 4.0` |

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-s, --source_path` | LangSplat data directory | `/workspace/data/my_project_langsplat` |
| `-m, --model_path` | 3DGS output directory | `/workspace/data/my_project/output` |
| `-f, --foundation_model` | Feature type | `langsplat` or `dino` |
| `-o, --output` | CF3 output directory | `/workspace/data/my_project_cf3` |

### Training Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iteration` | -1 | Load checkpoint iteration (-1 = latest) |
| `--iterations` | 30000 | Total CF3 training iterations |
| `--skip_train` | False | Skip training cameras rendering |
| `--skip_test` | False | Skip test cameras rendering |
| `--save_iterations` | [] | List of iterations to save (e.g., `7000 15000 30000`) |
| `--quiet` | False | Suppress console output |

**Example:**
```bash
python compact_feature_field.py \
    -s /workspace/data/my_project_langsplat \
    -m /workspace/data/my_project/output \
    -f langsplat \
    -o /workspace/data/my_project_cf3 \
    --iterations 15000 \
    --save_iterations 5000 10000 15000
```

### Feature Processing Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--antialiasing` | False | Enable anti-aliasing for rendering |
| `--finetune_decoder` | False | Fine-tune autoencoder decoder during CF3 training |
| `--normalize_feature` | False | Normalize semantic features before compression |
| `--use_render_feature` | False | Use rendered features as training target |
| `--use_gt_feature` | False | Use ground truth features as training target |
| `--filter_var` | False | Filter features by variance before compression |

**Recommended Combinations:**
```bash
# Basic: Fast training, good quality
--antialiasing --normalize_feature

# High quality: Better reconstruction
--antialiasing --finetune_decoder --normalize_feature --use_render_feature

# Maximum quality: Slowest but best results
--antialiasing --finetune_decoder --normalize_feature --use_render_feature --filter_var
```

### Memory Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--memory` | 96g | Docker container RAM limit |
| `--memory-swap` | 96g | Swap limit (set equal to RAM to disable swap) |
| `--shm-size` | 64gb | Shared memory for IPC |
| `--max_images` | -1 | Max training images to load (-1 = all) |
| `--image_downsample_factor` | 1.0 | Downsample factor (2.0 = half size, 4.0 = quarter) |
| `--skip_every_n` | 1 | Use every N-th image (2 = every other image) |
| `--compress_batch_size` | 64 | Autoencoder training batch size (reduce if OOM) |

**Docker Memory Settings:**
- **128GB RAM**: `--memory=96g --shm-size=64gb`
- **64GB RAM**: `--memory=48g --shm-size=32gb`
- **32GB RAM**: `--memory=24g --shm-size=16gb`

**Batch Size Guidelines:**
- **High RAM (96GB+)**: `--compress_batch_size 64` (default)
- **Medium RAM (48-96GB)**: `--compress_batch_size 32`
- **Low RAM (24-48GB)**: `--compress_batch_size 16`
- **Very Low RAM (<24GB)**: `--compress_batch_size 8`

**Image Loading Strategies:**

These parameters help reduce memory usage during Scene loading:

**`--max_images N`**: Limits total number of images loaded
- Use for testing with subset of data
- Example: `--max_images 50` loads only first 50 images

**`--skip_every_n N`**: Uses every N-th image
- Preserves spatial distribution of views
- Example: `--skip_every_n 2` uses 50% of images (every other image)
- Example: `--skip_every_n 3` uses 33% of images (every third image)

**`--image_downsample_factor F`**: Downsamples image resolution
- Reduces memory per image significantly
- Example: `--image_downsample_factor 2.0` = half resolution (75% less memory per image)
- Example: `--image_downsample_factor 4.0` = quarter resolution (94% less memory per image)

**`--compress_batch_size N`**: Controls autoencoder training batch size
- Reduce if you get OOM during STEP 2 (autoencoder training)
- Example: `--compress_batch_size 32` for limited memory
- Example: `--compress_batch_size 16` for very limited memory

**Combine strategies for maximum savings:**
```bash
# Example: Every other image at half resolution + small batch size
--skip_every_n 2 --image_downsample_factor 2.0 --compress_batch_size 32
```

**Memory Impact Estimation:**

| Images | Resolution | Est. RAM | With Downsampling (2x) | With Skip (2x) |
|--------|-----------|----------|------------------------|----------------|
| 100 | 4K (3840×2160) | ~60 GB | ~15 GB | ~30 GB |
| 100 | 2K (1920×1080) | ~30 GB | ~7.5 GB | ~15 GB |
| 50 | 4K (3840×2160) | ~30 GB | ~7.5 GB | - |
| 200 | 1K (1280×720) | ~20 GB | ~5 GB | ~10 GB |

**Automatic Batch Sizing:**
- CF3 automatically adjusts batch size based on available memory
- Initial batch size estimated: `(Available RAM × 0.6) / 1.5GB per camera`
- If OOM occurs, batch size is halved automatically
- Minimum batch size: 1

### Autoencoder Compression

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--compress_epoch` | 30 | Autoencoder training epochs |
| `--compress_batch_size` | 64 | Batch size for autoencoder training |
| `--ae_lr` | 0.0001 | Autoencoder learning rate |
| `--lambda_cossim` | 0.01 | Cosine similarity loss weight |
| `--lambda_metric` | 0.01 | Metric preservation loss weight |

**Tuning Tips:**
- **Faster compression**: `--compress_epoch 20 --compress_batch_size 128`
- **Better quality**: `--compress_epoch 50 --compress_batch_size 32`
- **Low memory**: `--compress_batch_size 16` (or even `8` for very low RAM)
- **OOM during STEP 2**: Reduce `--compress_batch_size` to 32 or 16

### Gaussian Filtering Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--contrib_threshold` | 0.0001 | Min contribution to keep Gaussians |
| `--alpha_threshold` | 0.001 | Min opacity to keep Gaussians |
| `--similarity_threshold` | 0.9 | Min feature similarity for merging (0-1) |
| `--merge_grad_threshold` | 0.0001 | Max gradient norm for merge candidates |

**Threshold Effects:**
- **Lower contrib_threshold** → More Gaussians kept → Higher quality, slower
- **Higher contrib_threshold** → Fewer Gaussians → More compact, faster
- **Higher similarity_threshold** → Stricter merging → More Gaussians
- **Lower similarity_threshold** → Aggressive merging → Fewer Gaussians

**Recommended Presets:**

**High Quality (Slow):**
```bash
--contrib_threshold 0.00005 \
--alpha_threshold 0.0005 \
--similarity_threshold 0.95 \
--merge_grad_threshold 0.00005
```

**Balanced (Default):**
```bash
--contrib_threshold 0.0001 \
--alpha_threshold 0.001 \
--similarity_threshold 0.9 \
--merge_grad_threshold 0.0001
```

**Fast/Compact:**
```bash
--contrib_threshold 0.0005 \
--alpha_threshold 0.005 \
--similarity_threshold 0.85 \
--merge_grad_threshold 0.0005
```

### Adaptive Sparsification

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--merge_interval` | 100 | How often to perform merging |
| `--merge_until_iter` | 25000 | Stop merging after this iteration |
| `--pruning` | True | Enable Gaussian pruning |
| `--merging` | True | Enable Gaussian merging |

**Sparsification Strategy:**
- **Aggressive**: `--merge_interval 50 --merge_until_iter 20000`
- **Conservative**: `--merge_interval 200 --merge_until_iter 28000`
- **No sparsification**: `--merge_interval 999999`

### Learning Rates

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--position_lr_init` | 0.00016 | Initial position learning rate |
| `--position_lr_final` | 0.0000016 | Final position learning rate |
| `--feature_lr` | 0.0025 | Feature learning rate |
| `--opacity_lr` | 0.05 | Opacity learning rate |
| `--scaling_lr` | 0.005 | Scaling learning rate |
| `--rotation_lr` | 0.001 | Rotation learning rate |
| `--gamma` | 0.33 | LR scheduler decay factor |

---

## CF3 Pipeline Stages

### Stage 1: Feature Accumulation (STEP 1)
**Duration:** ~1-5 minutes (depends on image count)

**What it does:**
1. Loads all camera views from Scene
2. Processes images in **adaptive batches** to manage RAM
3. Accumulates semantic features from each view onto 3D Gaussians
4. Computes feature variance for quality filtering

**Memory Management:**
- Automatically estimates safe batch size based on available RAM
- Monitors memory every 10 cameras
- If OOM occurs, reduces batch size by half and retries
- Outputs detailed memory statistics

**Console Output:**
```
============================================================
Memory Status:
  Total RAM: 128.00 GB
  Available RAM: 95.23 GB
  Current Usage: 12.45 GB
============================================================

Loading scene...
⚠️  IMPORTANT: Scene Loading Memory Warning
============================================================
Scene loading will load ALL cameras and images into RAM.
...

✓ Scene loaded successfully!
  Memory used by scene: 18.34 GB
  Current total usage: 30.79 GB
  Available memory: 76.89 GB

Total training cameras: 156

============================================================
Feature Accumulation Configuration:
  Available memory: 76.89 GB
  Estimated batch size: 30
  Total batches: 6
============================================================

Processing cameras 1-30/156 (batch size: 30)
Batch 1: 100%|████████████| 30/30 [01:24<00:00,  2.81s/it]
  Batch completed. Memory: 30.79 GB → 38.12 GB

Processing cameras 31-60/156 (batch size: 30)
Batch 2: 100%|████████████| 30/30 [01:22<00:00,  2.74s/it]
  Warning: Low memory - 4.23 GB available
  Batch completed. Memory: 38.12 GB → 42.67 GB

============================================================
Feature accumulation completed!
  Time elapsed: 512.34 seconds
  FPS: 0.30
  Final memory usage: 42.67 GB
============================================================
```

**Batch Size Calculation:**
```python
# Safety factor 60% of available memory
usable_memory = available_memory_gb * 0.6
# Estimated 1.5GB per high-res image with features
batch_size = usable_memory / 1.5
```

### Stage 2: Autoencoder Compression (STEP 2)
**Duration:** ~3-5 minutes (20-30 epochs)

**What it does:**
1. Filters Gaussians by contribution threshold
2. Optionally filters by feature variance (if `--filter_var`)
3. Trains autoencoder to compress 512D → 3D latent features
4. Preserves semantic relationships via cosine similarity

**Console Output:**
```
Training the autoencoder to compress semantic features...
Epoch 5/30: 100%|████████| 1250/1250 [00:08<00:00, 145.23it/s]
  Total Loss: 0.01234 (MSE: 0.00891, Cosine: 0.00234, Metric: 0.00109)
  
Epoch 10/30: 100%|███████| 1250/1250 [00:08<00:00, 147.89it/s]
  Total Loss: 0.00567 (MSE: 0.00412, Cosine: 0.00101, Metric: 0.00054)
```

**Loss Components:**
- **MSE Loss**: Feature reconstruction accuracy
- **Cosine Loss**: Semantic direction preservation
- **Metric Loss**: Pairwise similarity preservation

### Stage 3: CF3 Training (STEP 3)
**Duration:** ~10-20 minutes (30000 iterations)

**What it does:**
1. Trains compact feature field with adaptive sparsification
2. Periodically merges similar Gaussians
3. Prunes low-contribution Gaussians
4. Optionally fine-tunes decoder

**Console Output:**
```
Training progress: 15%|████▊              | 4500/30000 [02:34<14:21, 29.59it/s]
Loss: 0.0123456

xyz_grad: 0.0234, feature grad: 0.0156
Found matches: 2341, Reject cos: 234 pairs: 2107 2107

Training progress: 100%|███████████████| 30000/30000 [15:23<00:00, 32.47it/s]
```

**Merging Logic:**
- Every `--merge_interval` iterations
- Only merges Gaussians with gradient < `--merge_grad_threshold`
- Checks feature similarity (cosine > `--similarity_threshold`)
- Computes merged position, opacity, rotation, scaling

### Stage 4: Final Rendering
**Duration:** ~5-15 minutes (depends on image count and batch size)

**What it does:**
1. Renders test/train views with compact feature field
2. Computes FPS and quality metrics
3. Saves feature visualizations (PCA)
4. Generates comparison images

**Memory Management:**
- Render batch size: `max(1, min(10, available_memory / 3))`
- FPS batch size: `max(1, min(20, available_memory / 2))`

**Console Output:**
```
============================================================
Final rendering configuration:
  Available memory: 68.34 GB
  Render batch size: 10
  FPS collection batch size: 20
============================================================

Rendering and saving features with batch size: 10
  Rendering views 1-10/156
Render Batch 1: 100%|████████| 10/10 [00:45<00:00,  4.52s/it]
    Memory after batch: 52.34 GB

Collecting FPS importance with batch size: 20
  Processing views 1-20/156
FPS Batch 1: 100%|██████████| 20/20 [00:38<00:00,  1.92s/it]
```

---

## Performance Optimization

### Memory Optimization

**Problem: OOM during feature accumulation**
```bash
# Solution 1: Reduce initial batch size estimate
# Edit estimate_camera_batch_size() safety_factor
safety_factor=0.5  # More conservative (default: 0.6)

# Solution 2: Increase memory limit
docker run --memory=120g --memory-swap=120g ...

# Solution 3: Process fewer images at once
# The script will automatically reduce batch size on OOM
```

**Problem: OOM during autoencoder training**
```bash
# Solution: Reduce batch size
python compact_feature_field.py ... --compress_batch_size 32
# Or even lower:
python compact_feature_field.py ... --compress_batch_size 16
```

**Problem: OOM during rendering**
```bash
# The script automatically adjusts render batch size
# But you can monitor memory and kill other processes:
watch -n 1 nvidia-smi
htop
```

### Speed Optimization

**Fast Training (Development/Testing):**
```bash
python compact_feature_field.py \
    -s ... -m ... -f langsplat -o ... \
    --iterations 5000 \
    --compress_epoch 10 \
    --merge_interval 200 \
    --skip_test
```
**Duration:** ~8-12 minutes

**Balanced (Recommended):**
```bash
python compact_feature_field.py \
    -s ... -m ... -f langsplat -o ... \
    --iterations 15000 \
    --compress_epoch 20 \
    --antialiasing \
    --normalize_feature
```
**Duration:** ~15-20 minutes

**High Quality (Production):**
```bash
python compact_feature_field.py \
    -s ... -m ... -f langsplat -o ... \
    --iterations 30000 \
    --compress_epoch 30 \
    --antialiasing \
    --finetune_decoder \
    --normalize_feature \
    --use_render_feature \
    --filter_var
```
**Duration:** ~25-35 minutes

### Quality vs Size Trade-off

| Configuration | Gaussians | Quality | Training Time |
|--------------|-----------|---------|---------------|
| Aggressive Compression | ~10K | Good | ~10 min |
| Balanced | ~30K | Very Good | ~18 min |
| High Quality | ~50K | Excellent | ~30 min |
| No Compression | ~100K+ | Best | ~45 min |

**Aggressive Compression:**
```bash
--contrib_threshold 0.0005 \
--similarity_threshold 0.85 \
--merge_interval 50
```

**No Compression (Baseline):**
```bash
--merge_interval 999999 \
--skip_test \
--skip_train
```

---

## Troubleshooting CF3

### Issue 1: Scene Loading OOM
```
ERROR: Failed to load scene!
Insufficient memory to load all cameras/images.
Available memory: 45.23 GB
```

**Root Cause:** Scene class loads ALL images into RAM at once

**Solutions (in order of preference):**

**1. Limit number of images (fastest):**
```bash
# Use only first 50 images
python compact_feature_field.py ... --max_images 50

# Use only first 100 images  
python compact_feature_field.py ... --max_images 100
```

**2. Use subset of images (preserve distribution):**
```bash
# Use every 2nd image (50% reduction)
python compact_feature_field.py ... --skip_every_n 2

# Use every 3rd image (66% reduction)
python compact_feature_field.py ... --skip_every_n 3
```

**3. Downsample images (reduce memory per image):**
```bash
# Half resolution (75% memory reduction per image)
python compact_feature_field.py ... --image_downsample_factor 2.0

# Quarter resolution (93.75% memory reduction per image)
python compact_feature_field.py ... --image_downsample_factor 4.0
```

**4. Combine strategies (maximum savings):**
```bash
# Use every other image at half resolution
python compact_feature_field.py ... \
    --skip_every_n 2 \
    --image_downsample_factor 2.0

# Use first 50 images at quarter resolution
python compact_feature_field.py ... \
    --max_images 50 \
    --image_downsample_factor 4.0
```

**5. Free up RAM (traditional approach):**
```bash
# Check current memory
free -h

# Kill memory-heavy processes
pkill chrome
pkill firefox

# Then retry CF3
```

**Expected Memory Savings:**

| Strategy | Images | Memory Reduction |
|----------|--------|------------------|
| `--max_images 50` | 200 → 50 | 75% |
| `--skip_every_n 2` | 200 → 100 | 50% |
| `--skip_every_n 3` | 200 → 67 | 66% |
| `--image_downsample_factor 2.0` | 200 → 200 | ~75% (per image) |
| `--image_downsample_factor 4.0` | 200 → 200 | ~94% (per image) |
| Combined (skip_every_n=2 + downsample=2.0) | 200 → 100 | ~87.5% total |

### Issue 2: Feature Accumulation OOM
```
Out of memory! Reducing batch size...
  Current batch size: 30
  New batch size: 15
```

**Root Cause:** Images too large or too many features

**What happens:**
- Script automatically halves batch size
- Retries the same batch
- If batch size reaches 1 and still OOM → Fatal error

**Prevention:**
```python
# Lower safety factor in estimate_camera_batch_size()
safety_factor=0.4  # vs default 0.6
```

### Issue 3: cuSOLVER Error During Merging
```
[WARNING] cuSOLVER error encountered, skipping merge at iteration 5000.
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
```

**Root Cause:** Numerical instability in covariance matrix eigendecomposition

**Impact:** Merge skipped for that iteration, training continues

**Solutions:**
```bash
# 1. Reduce similarity threshold (less strict merging)
--similarity_threshold 0.85  # vs default 0.9

# 2. Increase merge interval (less frequent merging)
--merge_interval 200  # vs default 100

# 3. Add regularization (already implemented)
eps = 1e-6  # Added to diagonal
```

**Note:** This is a known CUDA numerical issue, not a critical error

### Issue 4: Low FPS After CF3
```
light fps: 12.34
initial fps: 45.67
```

**Root Cause:** Too many compact Gaussians remain

**Solutions:**
```bash
# Increase compression thresholds
--contrib_threshold 0.0005 \  # vs 0.0001
--similarity_threshold 0.85    # vs 0.9
```

### Issue 5: Poor Feature Quality
```
cossim: 0.654  # Should be > 0.85
l1: 0.234      # Should be < 0.1
```

**Root Cause:** Over-compression or insufficient training

**Solutions:**
```bash
# 1. Train longer
--iterations 40000  # vs 30000
--compress_epoch 40  # vs 30

# 2. Fine-tune decoder
--finetune_decoder

# 3. Use rendered features
--use_render_feature

# 4. Don't filter variance
# Remove --filter_var flag
```

---

## Output Files Explained

### Directory Structure
```
data/my_project_cf3/
├── train_before_merging/
│   └── gaussian_data.txt          # Stats before sparsification
├── train/                          # After full CF3 training
│   └── ours_30000/
│       ├── feature_vis/            # PCA visualizations
│       │   ├── 00000_feature_vis_concat_sub.png
│       │   ├── 00000_gt_feature_vis.png
│       │   ├── 00000_rendered_feature_vis.png
│       │   └── 00000_light_feature_vis.png
│       ├── saved_feature/          # Compressed features
│       │   └── 00000_fmap_CxHxW.pt
│       └── renders/                # RGB renders
│           └── 00000.png
└── point_cloud/
    ├── iteration_0/                # After autoencoder
    │   └── feature_field.ply
    └── iteration_30000/            # Final output
        ├── feature_field.ply       # Compact feature Gaussians
        ├── point_cloud.ply         # RGB Gaussians
        ├── gaussian_data.txt       # Statistics
        └── imp_score.npz           # Importance scores
```

### gaussian_data.txt
```
number of gaussians: 123456
number of compact feature gaussians: 34567
light fps: 28.45
initial fps: 18.23
```

**Interpretation:**
- **RGB Gaussians:** 123,456 (from 3DGS)
- **Feature Gaussians:** 34,567 (after CF3 compression)
- **Compression ratio:** 3.57x
- **Light FPS:** 28.45 (CF3 rendering speed)
- **Initial FPS:** 18.23 (original feature rendering)
- **Speedup:** 1.56x

### Feature Visualizations

**PCA Visualization Files:**
- `*_gt_feature_vis.png`: Ground truth semantic features
- `*_rendered_feature_vis.png`: Full FeatureGS (no compression)
- `*_light_feature_vis.png`: CF3 compressed features
- `*_feature_vis_concat_sub.png`: Side-by-side comparison

**Interpretation:**
- Similar colors → Good semantic preservation
- Cosine similarity > 0.85 → Excellent quality
- L1 loss < 0.1 → Good reconstruction

### PLY Files

**feature_field.ply:**
- Contains compact 3D latent features
- Size: Much smaller than original features
- Use: Semantic queries, object selection

**point_cloud.ply:**
- Contains RGB Gaussians (unchanged from 3DGS)
- Use: Visualization, rendering

---

## Best Practices

### 1. Memory Management
```bash
# Always specify memory limits
docker run --memory=96g --memory-swap=96g ...

# Monitor memory during training
watch -n 1 'free -h && nvidia-smi'

# Use appropriate batch sizes for your RAM
# 128GB RAM → batch_size 64
# 64GB RAM → batch_size 32
# 32GB RAM → batch_size 16
```

**Image Loading Strategies:**

```bash
# Strategy 1: Quick testing with subset
python compact_feature_field.py ... \
    --max_images 20 \
    --compress_epoch 10 \
    --iterations 3000

# Strategy 2: Limited RAM (64GB)
python compact_feature_field.py ... \
    --skip_every_n 2 \
    --image_downsample_factor 2.0 \
    --compress_batch_size 32

# Strategy 3: Very limited RAM (32GB)
python compact_feature_field.py ... \
    --max_images 50 \
    --image_downsample_factor 4.0 \
    --compress_batch_size 16

# Strategy 4: Production (128GB RAM)
python compact_feature_field.py ... \
    --image_downsample_factor 1.0 \  # Original size
    --compress_batch_size 64
```

**Memory Estimation Formula:**
```
Required RAM ≈ (num_images × image_resolution × feature_size) / downsample_factor²

Example for 100 images at 4K resolution:
- Original: 100 × 4K × 512D ≈ 60 GB
- With downsample=2.0: 60 / 4 ≈ 15 GB
- With downsample=4.0: 60 / 16 ≈ 3.75 GB
- With skip_every_n=2: 60 / 2 = 30 GB
```

### 2. Iterative Development
```bash
# Start with fast settings for testing
python compact_feature_field.py ... \
    --iterations 5000 \
    --compress_epoch 10 \
    --skip_test

# Then scale up for production
python compact_feature_field.py ... \
    --iterations 30000 \
    --compress_epoch 30 \
    --antialiasing --finetune_decoder
```

### 3. Monitor Progress
```bash
# Watch tensorboard (if enabled)
tensorboard --logdir CF3/runs

# Check intermediate outputs
ls -lh data/my_project_cf3/point_cloud/

# Verify feature quality periodically
python -c "
import torch
feat = torch.load('data/.../saved_feature/00000_fmap_CxHxW.pt')
print(f'Feature shape: {feat.shape}')
print(f'Feature range: [{feat.min():.3f}, {feat.max():.3f}]')
print(f'Feature mean: {feat.mean():.3f}')
"
```

### 4. Checkpoint Management
```bash
# Save checkpoints at multiple iterations
--save_iterations 5000 10000 15000 20000 30000

# Resume from specific checkpoint
--iteration 15000

# Compare different checkpoints
for iter in 5000 10000 15000; do
    echo "Iteration $iter:"
    cat data/my_project_cf3/point_cloud/iteration_$iter/gaussian_data.txt
done
```

---

**Performance:**
- Feature accumulation: Variable (depends on batch size and RAM)
- Autoencoder training: ~3-5 minutes
- CF3 training: ~10-20 minutes (30000 iterations)
- Final rendering: ~5-15 minutes
- **Total**: ~20-40 minutes
- Memory: Adaptive (8-96 GB RAM, 6-12 GB VRAM)

### Verify Phase 3 Output
```bash
ls -la data/${PROJECT_NAME}_cf3/
# Should see: train_before_merging/, train_after_merging/, point_cloud/

ls -la data/${PROJECT_NAME}_cf3/point_cloud/
# Should see: iteration_* directories

ls -la data/${PROJECT_NAME}_cf3/point_cloud/iteration_3000/
# Should see: point_cloud.ply, feature_field.ply
```

---

## Complete Pipeline Script

### Automated Full Pipeline
```bash
#!/bin/bash
# run_full_pipeline.sh
# Complete 3D reconstruction pipeline automation

set -e  # Exit on error

# Configuration
export PROJECT_NAME="${PROJECT_NAME:-my_project}"
export ITERATIONS="${ITERATIONS:-7000}"
export CF3_ITERATIONS="${CF3_ITERATIONS:-3000}"

echo "======================================"
echo "3D Reconstruction Pipeline"
echo "======================================"
echo "Project: ${PROJECT_NAME}"
echo "3DGS Iterations: ${ITERATIONS}"
echo "CF3 Iterations: ${CF3_ITERATIONS}"
echo "======================================"
echo ""

# Verify input directory exists
if [ ! -d "data/${PROJECT_NAME}/input" ]; then
    echo "ERROR: Input directory not found: data/${PROJECT_NAME}/input"
    echo "Please create the directory and add images:"
    echo "  mkdir -p data/${PROJECT_NAME}/input"
    echo "  cp /path/to/images/*.jpg data/${PROJECT_NAME}/input/"
    exit 1
fi

# Count images
IMAGE_COUNT=$(ls data/${PROJECT_NAME}/input/*.{jpg,png,JPG,PNG} 2>/dev/null | wc -l)
echo "Found ${IMAGE_COUNT} images in data/${PROJECT_NAME}/input/"
echo ""

if [ ${IMAGE_COUNT} -lt 3 ]; then
    echo "ERROR: At least 3 images are required for reconstruction"
    exit 1
fi

# Phase 1: COLMAP + 3DGS
echo "========================================="
echo "Phase 1: COLMAP + 3D Gaussian Splatting"
echo "========================================="
echo ""

docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=64gb \
    -v $(pwd)/data:/workspace/data \
    -e PROJECT_NAME="${PROJECT_NAME}" \
    -e ITERATIONS="${ITERATIONS}" \
    3d-reconstruction-pipeline:latest \
    bash -c "
cd /workspace/data/\${PROJECT_NAME}

echo 'Step 1/4: COLMAP Feature Extraction'
rm -f database.db
rm -rf sparse distorted

colmap feature_extractor \
    --database_path database.db \
    --image_path input \
    --SiftExtraction.use_gpu 0

echo ''
echo 'Step 2/4: COLMAP Feature Matching'
colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 0

echo ''
echo 'Step 3/4: COLMAP Sparse Reconstruction'
mkdir -p sparse

colmap mapper \
    --database_path database.db \
    --image_path input \
    --output_path sparse

if [ ! -d 'sparse/0' ]; then
    mkdir -p sparse/0
    if ls sparse/*.bin 1> /dev/null 2>&1; then
        mv sparse/*.bin sparse/0/
        mv sparse/*.txt sparse/0/ 2>/dev/null || true
    fi
fi

echo ''
echo 'Step 4/4: COLMAP Image Undistortion'
if [ -f 'sparse/0/cameras.bin' ]; then
    colmap image_undistorter \
        --image_path input \
        --input_path sparse/0 \
        --output_path distorted \
        --output_type COLMAP
else
    echo 'ERROR: COLMAP reconstruction failed'
    exit 1
fi

echo ''
echo '3DGS: Converting COLMAP data'
cd /workspace/gaussian-splatting
python convert.py -s /workspace/data/\${PROJECT_NAME}

echo ''
echo '3DGS: Training (\${ITERATIONS} iterations)'
python train.py \
    -s /workspace/data/\${PROJECT_NAME} \
    -m /workspace/data/\${PROJECT_NAME}/output \
    --iterations \${ITERATIONS} \
    --checkpoint_iterations \${ITERATIONS}
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Phase 1 failed"
    exit 1
fi

echo ""
echo "Phase 1 completed successfully!"
echo ""

# Phase 2: LangSplat
echo "========================================="
echo "Phase 2: LangSplat Preprocessing"
echo "========================================="
echo ""

docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=16gb \
    -v $(pwd)/data:/workspace/data \
    -e PROJECT_NAME="${PROJECT_NAME}" \
    3d-reconstruction-pipeline:latest \
    bash -c "
LANGSPLAT_DIR=\"/workspace/data/\${PROJECT_NAME}_langsplat\"

echo 'Step 1/2: Preparing directory structure'
rm -rf \${LANGSPLAT_DIR}
mkdir -p \${LANGSPLAT_DIR}/images

find /workspace/data/\${PROJECT_NAME}/images -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPG' -o -name '*.PNG' \) -exec cp {} \${LANGSPLAT_DIR}/images/ \;

cp -r /workspace/data/\${PROJECT_NAME}/sparse \${LANGSPLAT_DIR}/
cp -r /workspace/data/\${PROJECT_NAME}/output \${LANGSPLAT_DIR}/

echo ''
echo 'Step 2/2: Extracting semantic features'
cd /workspace/LangSplat

python preprocess.py \
    --dataset_path \${LANGSPLAT_DIR}
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Phase 2 failed"
    exit 1
fi

echo ""
echo "Phase 2 completed successfully!"
echo ""

# Phase 3: CF3
echo "========================================="
echo "Phase 3: CF3 (Compact Feature Field)"
echo "========================================="
echo ""

docker run -it --rm \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=16gb \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/CF3:/workspace/CF3:rw \
    -e PROJECT_NAME="${PROJECT_NAME}" \
    -e CF3_ITERATIONS="${CF3_ITERATIONS}" \
    3d-reconstruction-pipeline:latest \
    bash -c "
LANGSPLAT_DIR=\"/workspace/data/\${PROJECT_NAME}_langsplat\"
CF3_OUTPUT=\"/workspace/data/\${PROJECT_NAME}_cf3\"

echo 'Step 1/2: Converting features to CF3 format'
cd /workspace/CF3

python langsplat_feature_convert.py \
    --input_dir \${LANGSPLAT_DIR}/language_features \
    --output_dir \${LANGSPLAT_DIR}/language_feature_pt

echo ''
echo 'Step 2/2: Training CF3 (\${CF3_ITERATIONS} iterations)'
mkdir -p \${CF3_OUTPUT}

python compact_feature_field.py \
    -s \${LANGSPLAT_DIR} \
    -m \${LANGSPLAT_DIR}/output \
    --output \${CF3_OUTPUT} \
    --iteration \${CF3_ITERATIONS}
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Phase 3 failed"
    exit 1
fi

echo ""
echo "Phase 3 completed successfully!"
echo ""

# Summary
echo "======================================"
echo "Pipeline Completed Successfully!"
echo "======================================"
echo ""
echo "Output locations:"
echo "  Phase 1 (3DGS):      data/${PROJECT_NAME}/output/"
echo "  Phase 2 (LangSplat): data/${PROJECT_NAME}_langsplat/"
echo "  Phase 3 (CF3):       data/${PROJECT_NAME}_cf3/"
echo ""
echo "Final outputs:"
echo "  3D Gaussians:        data/${PROJECT_NAME}/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
echo "  Semantic features:   data/${PROJECT_NAME}_langsplat/language_feature_pt/"
echo "  Compact 3D scene:    data/${PROJECT_NAME}_cf3/point_cloud/iteration_${CF3_ITERATIONS}/"
echo ""
echo "To visualize:"
echo "  meshlab data/${PROJECT_NAME}_cf3/point_cloud/iteration_${CF3_ITERATIONS}/point_cloud.ply"
echo ""
```

---

## Environment Variables

### Phase 1 (COLMAP + 3DGS)
```bash
DATA_DIR="/workspace/data"                # Data root directory
PROJECT_NAME="my_scene"                   # Project name
COLMAP_USE_GPU=0                          # 0=CPU, 1=GPU (use 0 for headless)
```

### Phase 2 (LangSplat)
```bash
DATA_DIR="/workspace/data"
PROJECT_NAME="my_scene"
SAM_CHECKPOINT=""                         # Empty = auto-detect
# Or specify: SAM_CHECKPOINT="/workspace/data/checkpoints/sam_vit_b_01ec64.pth"
```

### Phase 3 (CF3)
```bash
DATA_DIR="/workspace/data"
PROJECT_NAME="my_scene"
FEATURE_TYPE="langsplat"                  # Feature type
CONTRIB_THRESHOLD=0.0001
ALPHA_THRESHOLD=0.001
SIMILARITY_THRESHOLD=0.9
MERGE_GRAD_THRESHOLD=0.0001
```

---

## Common Issues & Solutions

### Issue 1: COLMAP OpenGL Error
```
Error: Check failed: context_.create()
```
**Solution:** Use CPU mode with `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0`

### Issue 2: SAM Model Not Found
```
FileNotFoundError: No SAM checkpoint found in /workspace/data/checkpoints/
```
**Solution:** Download SAM model:
```bash
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Issue 3: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or image resolution:
```bash
# For LangSplat
python preprocess.py --dataset_path ... --resolution 1080

# For CF3, reduce shm-size
docker run ... --shm-size=8gb ...
```

### Issue 4: sparse/0 Directory Missing
```
RuntimeError: Could not find cameras.bin
```
**Solution:** Ensure sparse/0 is created after COLMAP:
```bash
mkdir -p data/${PROJECT_NAME}/sparse/0
cp data/${PROJECT_NAME}/distorted/sparse/*.bin data/${PROJECT_NAME}/sparse/0/
```

### Issue 5: Read-only File System Error (CF3)
```
OSError: [Errno 30] Read-only file system: 'runs'
```
**Solution:** Mount CF3 directory as read-write:
```bash
# Remove :ro flag
-v $(pwd)/CF3:/workspace/CF3  # Not :ro
```

### Issue 6: cuSOLVER Error (CF3)
```
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
```
**Solution:** This is a numerical instability issue in CF3. Currently being investigated. Try:
- Reduce `--similarity_threshold` to 0.85
- Reduce training iterations
- Use different random seed

---

## Performance Benchmarks

### Test System: Spark DGX with GB10 GPU
- **GPU**: NVIDIA GB10 (Grace Hopper), sm_121, 96GB VRAM
- **CUDA**: 13.0
- **Dataset**: carframe (26 images, 32 input → 26 reconstructed)

### Phase 1 Timing
- COLMAP (CPU): ~10 minutes
- 3DGS Training (7000 iter): ~7 minutes (14.29 it/s)
- **Total**: ~17 minutes

### Phase 2 Timing
- LangSplat (SAM ViT-B): ~5-7 minutes (26 images)
- LangSplat (SAM ViT-H): ~15-20 minutes (26 images)
- **Recommended**: Use ViT-B for faster processing

### Phase 3 Timing
- Feature conversion: ~1 minute
- Autoencoder training: ~3 minutes (20 epochs)
- CF3 training: ~10-15 minutes (3000 iterations)
- **Total**: ~14-19 minutes

### Full Pipeline
- **Total Duration**: ~36-43 minutes (with ViT-B)
- **Total Duration**: ~46-56 minutes (with ViT-H)

---

## Advanced Configuration

### Custom 3DGS Training
```bash
python train.py \
    -s /workspace/data/${PROJECT_NAME} \
    -m /workspace/data/${PROJECT_NAME}/output \
    --iterations 30000 \
    --position_lr_init 0.00016 \
    --position_lr_final 0.0000016 \
    --feature_lr 0.0025 \
    --opacity_lr 0.05 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    --densify_grad_threshold 0.0002 \
    --densification_interval 100 \
    --opacity_reset_interval 3000 \
    --lambda_dssim 0.2
```

### Custom CF3 Training
```bash
python compact_feature_field.py \
    -s /workspace/data/${PROJECT_NAME}_langsplat \
    -m /workspace/data/${PROJECT_NAME}_langsplat/output \
    -f langsplat \
    -o /workspace/data/${PROJECT_NAME}_cf3 \
    --iterations 5000 \
    --position_lr_init 0.00016 \
    --position_lr_final 0.0000016 \
    --feature_lr 0.0025 \
    --antialiasing \
    --finetune_decoder \
    --normalize_feature \
    --contrib_threshold 0.0001 \
    --alpha_threshold 0.001 \
    --similarity_threshold 0.9 \
    --merge_grad_threshold 0.0001 \
    --densify_grad_threshold 0.0002 \
    --densification_interval 100 \
    --opacity_reset_interval 500
```

---

## Quick Reference Commands

### Check GPU Status
```bash
nvidia-smi
docker run --rm --runtime=nvidia 3d-reconstruction-pipeline:latest nvidia-smi
```

### Monitor Training Progress
```bash
# 3DGS training logs
docker logs -f <container_id>

# CF3 tensorboard (if enabled)
docker run --rm -p 6006:6006 -v $(pwd)/CF3:/workspace/CF3 \
    3d-reconstruction-pipeline:latest \
    tensorboard --logdir /workspace/CF3/runs
```

### Clean Up Intermediate Files
```bash
# Remove intermediate COLMAP files
rm -rf data/${PROJECT_NAME}/distorted
rm data/${PROJECT_NAME}/database.db

# Remove temporary LangSplat directory
rm -rf data/${PROJECT_NAME}_langsplat

# Keep only final outputs
```

### Visualize Results
```bash
# Install MeshLab or CloudCompare to view .ply files
meshlab data/${PROJECT_NAME}_cf3/point_cloud/iteration_3000/point_cloud.ply
```

---

## Next Steps

1. **Visualization**: Use SIBR viewer or custom renderer to visualize results
2. **Fine-tuning**: Adjust hyperparameters for better quality
3. **Scaling**: Process larger datasets (100+ images)
4. **Integration**: Integrate with downstream applications

---

## Support & Troubleshooting

### Documentation Files
- `README.md`: Project overview
- `USAGE_GUIDE.md`: General usage guide
- `SAM_AUTO_DETECTION.md`: SAM model auto-detection details
- `PIPELINE_GUIDE.md`: This file

### Logs Location
- Phase 1: Console output + `data/${PROJECT_NAME}/output/events.out.tfevents.*`
- Phase 2: Console output
- Phase 3: Console output + `CF3/runs/` (if tensorboard enabled)

### Getting Help
1. Check error messages in console output
2. Review "Common Issues & Solutions" section
3. Verify directory structure matches expected layout
4. Check GPU memory usage with `nvidia-smi`

---

**Last Updated**: 2025-12-12  
**Pipeline Version**: 1.0  
**Compatible with**: CUDA 13.0, PyTorch 2.9.0a0, GB10 GPU
