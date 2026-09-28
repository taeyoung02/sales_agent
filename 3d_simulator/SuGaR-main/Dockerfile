# ==============================================================================
# SuGaR Dedicated Dockerfile - Hybrid Strategy (Final Fix)
# Base: CUDA 13.0 (GB10 GPU) + Miniconda Python 3.10 (Open3D Support)
# ==============================================================================

# 1. Base Image: CUDA 13.0, PyTorch 2.9 (System Level)
FROM nvcr.io/nvidia/pytorch:25.09-py3

# [CRITICAL FIX] Clear NVIDIA base image environment variables that conflict with Conda
# The base image sets PYTHONPATH to system Python paths, causing Conda packages to be ignored
ENV PYTHONPATH=""
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 2. Install System Packages
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Miniforge (Conda)
WORKDIR /tmp
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"

# 4. Create Python 3.10 Environment
RUN conda create -n sugar python=3.10 -y && \
    conda clean -afy

# 5. Install PyTorch with CUDA support (Clean Install)
# With PYTHONPATH cleared, no need for patchelf - conda environment will be used directly
RUN /opt/conda/envs/sugar/bin/pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 6. Install Other Dependencies
RUN /opt/conda/envs/sugar/bin/pip install --no-cache-dir \
    open3d \
    "numpy==1.26.4" \
    plyfile \
    tqdm \
    scipy \
    scikit-learn \
    trimesh \
    pyquaternion \
    fvcore \
    iopath \
    yacs \
    matplotlib \
    rich \
    cython

# 6.1. Install PyMCubes from source (to match current NumPy version)
RUN /opt/conda/envs/sugar/bin/pip install --no-cache-dir --no-binary pymcubes pymcubes==0.1.4

# 6.5. Install PyTorch3D (requires CUDA, must be after PyTorch installation)
# PyTorch3D is needed for mesh operations in SuGaR
# Build from source with specific version that works with PyTorch 2.x
# Install PyTorch3D from source - using main branch and removing conflicting make_float3
# Install to /opt/pytorch3d instead of /tmp for editable mode
RUN export CUDA_HOME=/usr/local/cuda && \
    export PATH=/usr/local/cuda/bin:/opt/conda/envs/sugar/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && \
    export FORCE_CUDA=1 && \
    export MAX_JOBS=2 && \
    cd /opt && \
    git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && \
    sed -i '70,74d' pytorch3d/csrc/pulsar/global.h && \
    /opt/conda/envs/sugar/bin/pip install --no-cache-dir --no-build-isolation -e .

# ===== 7. Copy Source Code (Submodules first for better caching) =====
WORKDIR /workspace

# Copy submodules separately - these change less frequently
COPY gaussian_splatting/submodules /workspace/SuGaR/gaussian_splatting/submodules

# 7.5. Prepare GLM for submodules (before building)
RUN if [ ! -f /workspace/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp ]; then \
        rm -rf /workspace/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/third_party/glm && \
        git clone --depth 1 https://github.com/g-truc/glm.git /workspace/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/third_party/glm; \
    fi

# ==============================================================================
# [FIX] Set CUDA Environment Variables (MUST be set BEFORE building CUDA extensions)
# ==============================================================================
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.1"
ENV FORCE_CUDA=1

# 8. Build SuGaR CUDA Submodules (Heavy compilation - cache this layer)
# Install directly to site-packages (not editable) for cleaner imports in Docker
RUN cd /workspace/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=/usr/local/cuda/bin:/opt/conda/envs/sugar/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && \
    export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.1" && \
    export FORCE_CUDA=1 && \
    /opt/conda/envs/sugar/bin/pip install . --no-build-isolation && \
    cd /workspace/SuGaR/gaussian_splatting/submodules/simple-knn && \
    /opt/conda/envs/sugar/bin/pip install . --no-build-isolation

# 9. Copy Application Code (Frequently changed - do this last)
WORKDIR /workspace/SuGaR
COPY . /workspace/SuGaR

# 9.5. Fix path bug in SuGaR cameras.py (missing slash)
RUN sed -i "s|gs_output_path + 'cameras.json'|os.path.join(gs_output_path, 'cameras.json')|g" sugar_scene/cameras.py && \
    sed -i 's|image_path = os.path.join(image_dir,  name + extension)|image_path = os.path.join(image_dir, name + extension if not name.endswith(extension) else name)|g' sugar_scene/cameras.py

# 9.6. Fix PyTorch 2.6+ weights_only=True default in torch.load
RUN sed -i 's/torch\.load(\([^,)]*\)\([,)]\)/torch.load(\1, weights_only=False\2/g' sugar_extractors/coarse_mesh.py && \
    sed -i 's/torch\.load(\([^,)]*\)\([,)]\)/torch.load(\1, weights_only=False\2/g' sugar_extractors/refined_mesh.py && \
    sed -i 's/torch\.load(\([^,)]*\)\([,)]\)/torch.load(\1, weights_only=False\2/g' sugar_scene/sugar_model.py && \
    sed -i 's/torch\.load(\([^,)]*\)\([,)]\)/torch.load(\1, weights_only=False\2/g' metrics.py

# 10. Final Setup
RUN echo "conda activate sugar" >> ~/.bashrc

# [CRITICAL] Runtime environment setup - conda environment takes precedence
ENV CONDA_DEFAULT_ENV=sugar
ENV PATH="/opt/conda/envs/sugar/bin:$PATH"
# Add SuGaR paths for module imports (root + gaussian_splatting)
# Do NOT append $PYTHONPATH to avoid system Python 3.12 path pollution
ENV PYTHONPATH="/workspace/SuGaR:/workspace/SuGaR/gaussian_splatting"
# Conda libs first, then CUDA, preventing base PyTorch lib conflicts
ENV LD_LIBRARY_PATH="/opt/conda/envs/sugar/lib:/usr/local/cuda/lib64"
ENV PYTHONNOUSERSITE=1

WORKDIR /workspace
CMD ["/bin/bash"]