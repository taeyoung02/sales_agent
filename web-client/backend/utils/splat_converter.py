"""
PLY to KSplat converter for 3D Gaussian Splatting
Converts PLY files (from 3DGS training) to KSplat binary format
"""

import struct
import numpy as np
from plyfile import PlyData
from pathlib import Path
from typing import Optional, Tuple
import os


class PLYToKSplatConverter:
    """Converter from PLY format to KSplat binary format"""

    def __init__(self):
        # KSplat format constants
        self.KSPLAT_VERSION = 1
        self.FLOAT_SIZE = 4  # 32-bit float
        self.INT_SIZE = 4  # 32-bit int

    def read_ply(self, ply_path: str) -> Tuple[np.ndarray, dict]:
        """
        Read PLY file and extract Gaussian splat data

        Args:
            ply_path: Path to PLY file

        Returns:
            Tuple of (gaussian_data, metadata)
            gaussian_data: numpy array with shape (N, 17) containing:
                [x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
                 opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3]
        """
        plydata = PlyData.read(ply_path)

        # Extract vertex data (Gaussian splats)
        # PlyElement의 data 속성이 numpy structured array입니다
        vertex_element = plydata["vertex"]
        vertex = vertex_element.data  # numpy structured array

        # Get required fields
        positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
            np.float32
        )

        # Get normals if available
        if "nx" in vertex.dtype.names:
            normals = np.stack(
                [vertex["nx"], vertex["ny"], vertex["nz"]], axis=1
            ).astype(np.float32)
        else:
            # Default normals (will be ignored in rendering)
            normals = np.zeros((len(positions), 3), dtype=np.float32)

        # Get spherical harmonics (DC components)
        if "f_dc_0" in vertex.dtype.names:
            sh_dc = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
            ).astype(np.float32)
        else:
            # Default white color
            sh_dc = np.ones((len(positions), 3), dtype=np.float32) * 0.5

        # Get opacity
        if "opacity" in vertex.dtype.names:
            opacity = vertex["opacity"].astype(np.float32)
        else:
            opacity = np.ones(len(positions), dtype=np.float32)

        # Get scale
        if "scale_0" in vertex.dtype.names:
            scale = np.stack(
                [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
            ).astype(np.float32)
        else:
            # Default scale
            scale = np.ones((len(positions), 3), dtype=np.float32) * 0.01

        # Get rotation (quaternion)
        if "rot_0" in vertex.dtype.names:
            rotation = np.stack(
                [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
                axis=1,
            ).astype(np.float32)
        else:
            # Default identity quaternion
            rotation = np.zeros((len(positions), 4), dtype=np.float32)
            rotation[:, 0] = 1.0  # w component

        # Combine all data
        # Format: [x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
        #          opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3]
        gaussian_data = np.concatenate(
            [positions, normals, sh_dc, opacity[:, None], scale, rotation], axis=1
        )

        metadata = {
            "num_gaussians": len(positions),
            "has_normals": "nx" in vertex.dtype.names,
            "has_sh": "f_dc_0" in vertex.dtype.names,
            "has_opacity": "opacity" in vertex.dtype.names,
            "has_scale": "scale_0" in vertex.dtype.names,
            "has_rotation": "rot_0" in vertex.dtype.names,
        }

        return gaussian_data, metadata

    def write_ksplat(
        self,
        gaussian_data: np.ndarray,
        output_path: str,
        metadata: Optional[dict] = None,
    ):
        """
        Write Gaussian splat data to KSplat binary format

        KSplat format:
        - Header (16 bytes):
          - version (int32): Format version
          - num_gaussians (int32): Number of Gaussian splats
          - reserved (int64): Reserved for future use
        - Data (per Gaussian, 68 bytes):
          - position (3 * float32): x, y, z
          - normal (3 * float32): nx, ny, nz (optional, can be zeros)
          - sh_dc (3 * float32): Spherical harmonics DC components (RGB)
          - opacity (float32): Opacity value
          - scale (3 * float32): Scale factors
          - rotation (4 * float32): Quaternion rotation (w, x, y, z)

        Args:
            gaussian_data: numpy array with shape (N, 17)
            output_path: Path to output KSplat file
            metadata: Optional metadata dictionary
        """
        num_gaussians = len(gaussian_data)

        with open(output_path, "wb") as f:
            # Write header
            f.write(struct.pack("i", self.KSPLAT_VERSION))  # version (4 bytes)
            f.write(struct.pack("i", num_gaussians))  # num_gaussians (4 bytes)
            f.write(struct.pack("q", 0))  # reserved (8 bytes)
            # Total header: 16 bytes

            # Write Gaussian data
            for i in range(num_gaussians):
                gaussian = gaussian_data[i]

                # Position (x, y, z) - 12 bytes
                f.write(struct.pack("fff", gaussian[0], gaussian[1], gaussian[2]))

                # Normal (nx, ny, nz) - 12 bytes
                f.write(struct.pack("fff", gaussian[3], gaussian[4], gaussian[5]))

                # Spherical harmonics DC (f_dc_0, f_dc_1, f_dc_2) - 12 bytes
                f.write(struct.pack("fff", gaussian[6], gaussian[7], gaussian[8]))

                # Opacity - 4 bytes
                f.write(struct.pack("f", gaussian[9]))

                # Scale (scale_0, scale_1, scale_2) - 12 bytes
                f.write(struct.pack("fff", gaussian[10], gaussian[11], gaussian[12]))

                # Rotation quaternion (rot_0, rot_1, rot_2, rot_3) - 16 bytes
                f.write(
                    struct.pack(
                        "ffff", gaussian[13], gaussian[14], gaussian[15], gaussian[16]
                    )
                )

                # Total per Gaussian: 68 bytes

    def convert(self, ply_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PLY file to KSplat format

        Args:
            ply_path: Path to input PLY file
            output_path: Path to output KSplat file (optional)

        Returns:
            Path to output KSplat file
        """
        ply_path = Path(ply_path)

        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Generate output path if not provided
        if output_path is None:
            output_path = ply_path.parent / f"{ply_path.stem}.ksplat"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read PLY file
        print(f"Reading PLY file: {ply_path}")
        gaussian_data, metadata = self.read_ply(str(ply_path))

        print(f"Found {metadata['num_gaussians']} Gaussian splats")
        print(f"Metadata: {metadata}")

        # Write KSplat file
        print(f"Writing KSplat file: {output_path}")
        self.write_ksplat(gaussian_data, str(output_path), metadata)

        file_size = output_path.stat().st_size
        print(
            f"Conversion complete! Output file size: {file_size / 1024 / 1024:.2f} MB"
        )

        return str(output_path)

    def convert_batch(
        self, input_dir: str, output_dir: Optional[str] = None, pattern: str = "*.ply"
    ):
        """
        Convert multiple PLY files in a directory

        Args:
            input_dir: Directory containing PLY files
            output_dir: Output directory (default: same as input)
            pattern: File pattern to match (default: "*.ply")
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        ply_files = list(input_dir.glob(pattern))

        if not ply_files:
            print(f"No PLY files found in {input_dir} matching pattern {pattern}")
            return

        print(f"Found {len(ply_files)} PLY files to convert")

        for ply_file in ply_files:
            try:
                output_file = output_dir / f"{ply_file.stem}.ksplat"
                self.convert(str(ply_file), str(output_file))
                print(f"✓ Converted: {ply_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"✗ Failed to convert {ply_file.name}: {str(e)}")


def convert_ply_to_ksplat(ply_path: str, output_path: Optional[str] = None) -> str:
    """
    Convenience function to convert PLY to KSplat

    Args:
        ply_path: Path to input PLY file
        output_path: Path to output KSplat file (optional)

    Returns:
        Path to output KSplat file
    """
    converter = PLYToKSplatConverter()
    return converter.convert(ply_path, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python splat_converter.py <ply_file> [output_file]")
        print("   or: python splat_converter.py <input_dir> [output_dir] --batch")
        sys.exit(1)

    converter = PLYToKSplatConverter()

    if "--batch" in sys.argv:
        # Batch conversion
        input_dir = sys.argv[1]
        output_dir = (
            sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "--batch" else None
        )
        converter.convert_batch(input_dir, output_dir)
    else:
        # Single file conversion
        ply_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        converter.convert(ply_path, output_path)
