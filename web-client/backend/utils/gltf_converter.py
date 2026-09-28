"""
PLY to glTF converter
Converts PLY files (mesh or point cloud) to glTF/GLB format
"""

import numpy as np
from plyfile import PlyData
from pathlib import Path
from typing import Optional, Tuple
import pygltflib
import base64
import json


class PLYToGLTFConverter:
    """Converter from PLY format to glTF/GLB format"""

    def __init__(self):
        pass

    def read_ply(self, ply_path: str) -> Tuple[dict, dict]:
        """
        Read PLY file and extract mesh/point cloud data

        Args:
            ply_path: Path to PLY file

        Returns:
            Tuple of (mesh_data, metadata)
            mesh_data: Dictionary containing vertices, faces, normals, colors
            metadata: Dictionary with file information
        """
        plydata = PlyData.read(ply_path)

        # Extract vertex data
        vertex_element = plydata["vertex"]
        vertex = vertex_element.data  # numpy structured array

        # Get positions (required)
        positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
            np.float32
        )

        # Get normals if available
        normals = None
        if "nx" in vertex.dtype.names:
            normals = np.stack(
                [vertex["nx"], vertex["ny"], vertex["nz"]], axis=1
            ).astype(np.float32)

        # Get colors if available
        # 3DGS PLY files use Spherical Harmonics (f_dc_0, f_dc_1, f_dc_2) for color
        colors = None
        if (
            "r" in vertex.dtype.names
            and "g" in vertex.dtype.names
            and "b" in vertex.dtype.names
        ):
            # Direct RGB values (0-255)
            colors = np.stack([vertex["r"], vertex["g"], vertex["b"]], axis=1).astype(
                np.uint8
            )
        elif (
            "red" in vertex.dtype.names
            and "green" in vertex.dtype.names
            and "blue" in vertex.dtype.names
        ):
            # Direct RGB values (0-255)
            colors = np.stack(
                [vertex["red"], vertex["green"], vertex["blue"]], axis=1
            ).astype(np.uint8)
        elif (
            "f_dc_0" in vertex.dtype.names
            and "f_dc_1" in vertex.dtype.names
            and "f_dc_2" in vertex.dtype.names
        ):
            # 3DGS Spherical Harmonics DC components (need to convert to RGB)
            # SH DC components are already in RGB space, but need sigmoid activation
            sh_dc = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
            ).astype(np.float32)

            # Apply sigmoid activation and convert to 0-255 range
            # SH values are typically in range [-1, 1] or similar
            # Sigmoid: 1 / (1 + exp(-x))
            rgb = 1.0 / (1.0 + np.exp(-sh_dc))
            # Clamp to [0, 1] and convert to [0, 255]
            rgb = np.clip(rgb, 0.0, 1.0)
            colors = (rgb * 255.0).astype(np.uint8)

        # Get faces if available (for mesh, not point cloud)
        faces = None
        if "face" in plydata:
            face_element = plydata["face"]
            face_data = face_element.data

            # PLY faces can be stored in different formats
            if "vertex_indices" in face_data.dtype.names:
                # List of lists format
                face_list = face_data["vertex_indices"]
                # Convert to flat array of triangles
                triangles = []
                for face in face_list:
                    if len(face) >= 3:
                        # Triangulate polygon (simple fan triangulation)
                        for i in range(1, len(face) - 1):
                            triangles.extend([face[0], face[i], face[i + 1]])
                if triangles:
                    faces = np.array(triangles, dtype=np.uint32)
            elif "vertex_index" in face_data.dtype.names:
                # Flat array format
                faces = face_data["vertex_index"].astype(np.uint32)

        mesh_data = {
            "positions": positions,
            "normals": normals,
            "colors": colors,
            "faces": faces,
        }

        metadata = {
            "num_vertices": len(positions),
            "num_faces": len(faces) // 3 if faces is not None else 0,
            "has_normals": normals is not None,
            "has_colors": colors is not None,
            "is_mesh": faces is not None,
            "is_point_cloud": faces is None,
        }

        return mesh_data, metadata

    def create_gltf(
        self,
        mesh_data: dict,
        output_path: str,
        metadata: Optional[dict] = None,
        format: str = "gltf",
    ):
        """
        Create glTF file from mesh data

        Args:
            mesh_data: Dictionary with positions, normals, colors, faces
            output_path: Path to output glTF/GLB file
            metadata: Optional metadata dictionary
            format: "gltf" or "glb" (default: "gltf")
        """
        positions = mesh_data["positions"]
        normals = mesh_data.get("normals")
        colors = mesh_data.get("colors")
        faces = mesh_data.get("faces")

        # Prepare buffer data in correct order
        buffer_data = bytearray()
        buffer_views = []
        accessors = []
        current_offset = 0

        # 1. Positions (required)
        positions_bytes = positions.tobytes()
        buffer_data.extend(positions_bytes)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0, byteOffset=current_offset, byteLength=positions.nbytes
            )
        )
        accessors.append(
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(positions),
                type=pygltflib.VEC3,
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            )
        )
        current_offset += positions.nbytes

        # 2. Normals (optional)
        normal_accessor_idx = None
        if normals is not None:
            normals_bytes = normals.tobytes()
            buffer_data.extend(normals_bytes)
            buffer_views.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=current_offset,
                    byteLength=normals.nbytes,
                )
            )
            normal_accessor_idx = len(accessors)
            accessors.append(
                pygltflib.Accessor(
                    bufferView=len(buffer_views) - 1,
                    componentType=pygltflib.FLOAT,
                    count=len(normals),
                    type=pygltflib.VEC3,
                    min=normals.min(axis=0).tolist(),
                    max=normals.max(axis=0).tolist(),
                )
            )
            current_offset += normals.nbytes

        # 3. Colors (optional)
        color_accessor_idx = None
        if colors is not None:
            colors_bytes = colors.tobytes()
            buffer_data.extend(colors_bytes)
            buffer_views.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=current_offset,
                    byteLength=colors.nbytes,
                )
            )
            color_accessor_idx = len(accessors)
            accessors.append(
                pygltflib.Accessor(
                    bufferView=len(buffer_views) - 1,
                    componentType=pygltflib.UNSIGNED_BYTE,
                    count=len(colors),
                    type=pygltflib.VEC3,
                    normalized=True,
                )
            )
            current_offset += colors.nbytes

        # 4. Indices (optional, for meshes)
        indices_accessor_idx = None
        if faces is not None:
            faces_bytes = faces.tobytes()
            buffer_data.extend(faces_bytes)
            buffer_views.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=current_offset,
                    byteLength=faces.nbytes,
                )
            )
            indices_accessor_idx = len(accessors)
            accessors.append(
                pygltflib.Accessor(
                    bufferView=len(buffer_views) - 1,
                    componentType=pygltflib.UNSIGNED_INT,
                    count=len(faces),
                    type=pygltflib.SCALAR,
                    min=[int(faces.min())],
                    max=[int(faces.max())],
                )
            )

        # Create glTF structure
        primitive_attributes = {"POSITION": 0}
        if normal_accessor_idx is not None:
            primitive_attributes["NORMAL"] = normal_accessor_idx
        if color_accessor_idx is not None:
            primitive_attributes["COLOR_0"] = color_accessor_idx

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=primitive_attributes,
                            indices=indices_accessor_idx,
                        )
                    ]
                )
            ],
            accessors=accessors,
            bufferViews=buffer_views,
            buffers=[],
        )

        # Create buffer
        if format == "glb":
            # GLB format: binary buffer embedded
            gltf.buffers = [pygltflib.Buffer(byteLength=len(buffer_data))]
        else:
            # glTF format: external buffer file
            buffer_filename = Path(output_path).stem + ".bin"
            buffer_path = Path(output_path).parent / buffer_filename

            # Write buffer file
            with open(buffer_path, "wb") as f:
                f.write(buffer_data)

            # Add buffer reference
            gltf.buffers = [
                pygltflib.Buffer(uri=buffer_filename, byteLength=len(buffer_data))
            ]

        # Save glTF file
        if format == "glb":
            # Save as GLB (binary)
            gltf.save_binary(output_path, buffer_data)
        else:
            # Save as glTF (JSON + external buffer)
            gltf.save(output_path)

    def convert(
        self,
        ply_path: str,
        output_path: Optional[str] = None,
        format: str = "gltf",
    ) -> str:
        """
        Convert PLY file to glTF/GLB format

        Args:
            ply_path: Path to input PLY file
            output_path: Path to output glTF/GLB file (optional)
            format: "gltf" or "glb" (default: "gltf")

        Returns:
            Path to output glTF/GLB file
        """
        ply_path = Path(ply_path)

        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Generate output path if not provided
        if output_path is None:
            if format == "glb":
                output_path = ply_path.parent / f"{ply_path.stem}.glb"
            else:
                output_path = ply_path.parent / f"{ply_path.stem}.gltf"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read PLY file
        print(f"Reading PLY file: {ply_path}")
        mesh_data, metadata = self.read_ply(str(ply_path))

        print(f"Found {metadata['num_vertices']} vertices")
        if metadata["is_mesh"]:
            print(f"Found {metadata['num_faces']} faces (mesh)")
        else:
            print("Point cloud (no faces)")

        # Create glTF file
        print(f"Writing {format.upper()} file: {output_path}")
        self.create_gltf(mesh_data, str(output_path), metadata, format)

        file_size = output_path.stat().st_size
        print(
            f"Conversion complete! Output file size: {file_size / 1024 / 1024:.2f} MB"
        )

        return str(output_path)

    def convert_batch(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.ply",
        format: str = "gltf",
    ):
        """
        Convert multiple PLY files in a directory

        Args:
            input_dir: Directory containing PLY files
            output_dir: Output directory (default: same as input)
            pattern: File pattern to match (default: "*.ply")
            format: "gltf" or "glb" (default: "gltf")
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
                if format == "glb":
                    output_file = output_dir / f"{ply_file.stem}.glb"
                else:
                    output_file = output_dir / f"{ply_file.stem}.gltf"
                self.convert(str(ply_file), str(output_file), format)
                print(f"✓ Converted: {ply_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"✗ Failed to convert {ply_file.name}: {str(e)}")


def convert_ply_to_gltf(
    ply_path: str, output_path: Optional[str] = None, format: str = "gltf"
) -> str:
    """
    Convenience function to convert PLY to glTF/GLB

    Args:
        ply_path: Path to input PLY file
        output_path: Path to output glTF/GLB file (optional)
        format: "gltf" or "glb" (default: "gltf")

    Returns:
        Path to output glTF/GLB file
    """
    converter = PLYToGLTFConverter()
    return converter.convert(ply_path, output_path, format)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gltf_converter.py <ply_file> [output_file] [--glb]")
        print(
            "   or: python gltf_converter.py <input_dir> [output_dir] --batch [--glb]"
        )
        sys.exit(1)

    converter = PLYToGLTFConverter()
    format = "glb" if "--glb" in sys.argv else "gltf"

    if "--batch" in sys.argv:
        # Batch conversion
        input_dir = sys.argv[1]
        output_dir = (
            sys.argv[2]
            if len(sys.argv) > 2 and sys.argv[2] not in ["--batch", "--glb"]
            else None
        )
        converter.convert_batch(input_dir, output_dir, format=format)
    else:
        # Single file conversion
        ply_path = sys.argv[1]
        output_path = (
            sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ["--glb"] else None
        )
        converter.convert(ply_path, output_path, format)
