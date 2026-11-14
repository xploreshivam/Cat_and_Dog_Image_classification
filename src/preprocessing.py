"""Image Resizing and Pixel Normalization Module """
from typing import Tuple
import numpy as np
from PIL import Image


def resize_and_normalize_image(
    pil_image: Image.Image, target_size: Tuple[int, int] = (150, 150)
) -> np.ndarray:
    """Resizes an RGB image and normalizes pixel values to [0.0, 1.0]."""
    resized = pil_image.convert("RGB").resize(target_size)
    normalized = np.array(resized, dtype="float32") / 255.0
    return np.expand_dims(normalized, axis=0)
