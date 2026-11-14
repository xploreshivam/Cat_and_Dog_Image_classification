"""Duplicate Image Check and Filtering Module """
import hashlib
from pathlib import Path


def compute_file_md5(filepath: Path) -> str:
    """Computes a 128-bit MD5 hash over raw file bytes to prevent split duplicates."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
