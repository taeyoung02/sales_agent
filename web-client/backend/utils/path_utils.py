"""
Utility functions for path resolution
"""

from pathlib import Path
import os


def get_source_file_path(path: str, source_root: Path) -> Path:
    """
    Get absolute path for a file in the source directory.
    Handles paths that may already include 'web-client/source/' prefix.

    Args:
        path: File path (can be absolute, relative to source root, or include web-client/source/)
        source_root: Path to the web-client/source directory

    Returns:
        Absolute Path object
    """
    if Path(path).is_absolute():
        return Path(path)

    # Remove web-client/source/ prefix if present
    path_str = str(path)
    if "web-client/source/" in path_str:
        relative_path = path_str.split("web-client/source/", 1)[1]
    elif path_str.startswith("source/"):
        relative_path = path_str.split("source/", 1)[1]
    else:
        relative_path = path_str

    return source_root / relative_path


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent.parent


def get_source_root() -> Path:
    """Get the web-client/source directory"""
    # Docker 환경에서는 /backend/source로 마운트됨
    if os.path.exists("/backend/source"):
        return Path("/backend/source")
    return get_project_root() / "web-client" / "source"
