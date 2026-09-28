#!/bin/bash
#
# SuGaR Mesh Extraction with Mask-based Voting Culling
# =====================================================
# This script demonstrates how to extract a mesh from a coarse SuGaR model
# with background Gaussian removal using mask-based voting.
#
# Usage:
#   ./extract_mesh_with_culling.sh <scene_name>
#
# Example:
#   ./extract_mesh_with_culling.sh volvo
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

SCENE_NAME="${1:-volvo}"  # Default to 'volvo' if not provided

# Paths
BASE_DIR="/home/kolon/Desktop/work/Pipeline"
SCENE_DIR="${BASE_DIR}/data/${SCENE_NAME}"
OUTPUT_DIR="${BASE_DIR}/output/${SCENE_NAME}"
SUGAR_DIR="${BASE_DIR}/SuGaR-main"

# Model paths
GS_CHECKPOINT="${OUTPUT_DIR}/point_cloud/iteration_7000"
SUGAR_CHECKPOINT="${OUTPUT_DIR}/coarse_sugar_7k.ckpt"
ITERATION=7000

# Mask settings
MASK_DIR="${SCENE_DIR}/masks"
ENABLE_CULLING="True"
VOTING_THRESHOLD=0.75
DILATION_KERNEL=7
DEBUG_MODE="True"

# Output settings
MESH_OUTPUT_DIR="${OUTPUT_DIR}/coarse_mesh"

# ============================================================================
# Validation
# ============================================================================

echo "=========================================="
echo "SuGaR Mesh Extraction with Mask Culling"
echo "=========================================="
echo ""

# Check if scene directory exists
if [ ! -d "${SCENE_DIR}" ]; then
    echo "ERROR: Scene directory not found: ${SCENE_DIR}"
    echo "Please provide a valid scene name."
    exit 1
fi

# Check if Gaussian Splatting checkpoint exists
if [ ! -d "${GS_CHECKPOINT}" ]; then
    echo "ERROR: Gaussian Splatting checkpoint not found: ${GS_CHECKPOINT}"
    echo "Please train vanilla 3DGS first:"
    echo "  python gaussian_splatting/train.py -s ${SCENE_DIR} -m ${OUTPUT_DIR} --iterations 7000"
    exit 1
fi

# Check if SuGaR checkpoint exists
if [ ! -f "${SUGAR_CHECKPOINT}" ]; then
    echo "ERROR: SuGaR checkpoint not found: ${SUGAR_CHECKPOINT}"
    echo "Please train coarse SuGaR first:"
    echo "  python train.py -s ${SCENE_DIR} -c ${GS_CHECKPOINT} -r density --iterations 7000"
    exit 1
fi

# Check if mask directory exists
if [ ! -d "${MASK_DIR}" ]; then
    echo "WARNING: Mask directory not found: ${MASK_DIR}"
    echo "Mask-based culling will be skipped."
    echo ""
    echo "To generate masks, run:"
    echo "  python generate_rembg_masks.py --input_dir ${SCENE_DIR}/images --output_dir ${MASK_DIR}"
    echo ""
    ENABLE_CULLING="False"
    MASK_DIR=""
fi

# ============================================================================
# Display configuration
# ============================================================================

echo "Configuration:"
echo "  Scene name:        ${SCENE_NAME}"
echo "  Scene directory:   ${SCENE_DIR}"
echo "  Output directory:  ${OUTPUT_DIR}"
echo "  GS checkpoint:     ${GS_CHECKPOINT}"
echo "  SuGaR checkpoint:  ${SUGAR_CHECKPOINT}"
echo "  Iteration:         ${ITERATION}"
echo ""
echo "Mask Culling Settings:"
echo "  Enabled:           ${ENABLE_CULLING}"
if [ "${ENABLE_CULLING}" = "True" ]; then
    echo "  Mask directory:    ${MASK_DIR}"
    echo "  Voting threshold:  ${VOTING_THRESHOLD}"
    echo "  Dilation kernel:   ${DILATION_KERNEL}"
    echo "  Debug mode:        ${DEBUG_MODE}"
fi
echo ""

# ============================================================================
# Run mesh extraction
# ============================================================================

cd "${SUGAR_DIR}"

echo "Starting mesh extraction..."
echo ""

# Build command
CMD="python extract_mesh.py \
    -s ${SCENE_DIR} \
    -c ${GS_CHECKPOINT} \
    -m ${SUGAR_CHECKPOINT} \
    -i ${ITERATION} \
    -o ${MESH_OUTPUT_DIR}"

# Add mask culling parameters if enabled
if [ "${ENABLE_CULLING}" = "True" ] && [ -n "${MASK_DIR}" ]; then
    CMD="${CMD} \
    --mask_dir ${MASK_DIR} \
    --enable_mask_culling ${ENABLE_CULLING} \
    --mask_voting_threshold ${VOTING_THRESHOLD} \
    --mask_dilation_kernel ${DILATION_KERNEL} \
    --debug_culling ${DEBUG_MODE}"
fi

# Display command
echo "Executing command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Mesh Extraction Complete!"
echo "=========================================="
echo ""
echo "Output location: ${MESH_OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${MESH_OUTPUT_DIR}"/*.ply 2>/dev/null || echo "  (no .ply files found)"
echo ""

if [ "${ENABLE_CULLING}" = "True" ] && [ "${DEBUG_MODE}" = "True" ]; then
    DEBUG_DIR="${MESH_OUTPUT_DIR}/culling_debug"
    if [ -d "${DEBUG_DIR}" ]; then
        echo "Debug outputs:"
        echo "  ${DEBUG_DIR}"
        echo ""
        echo "Culling statistics:"
        cat "${DEBUG_DIR}/culling_stats.json" 2>/dev/null || echo "  (stats file not found)"
        echo ""
    fi
fi

echo "Next steps:"
echo "  1. View mesh in MeshLab or Blender"
echo "  2. Refine mesh with: python refine_mesh.py ..."
echo "  3. Extract texture with: python extract_texture.py ..."
echo ""
