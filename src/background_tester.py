"""Testing on Real-World and Varied Backgrounds """
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf


def predict_custom_background_image(
    model: tf.keras.Model, image_path: str, image_size: Tuple[int, int] = (160, 160)
) -> Dict[str, Any]:
    """Runs single-image inference on indoor or outdoor pet photos."""
    img = Image.open(image_path).convert("RGB").resize(image_size)
    tensor = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    prob = float(model.predict(tensor, verbose=0)[0][0])
    label = "Dog" if prob >= 0.5 else "Cat"
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    return {"label": label, "confidence_pct": round(confidence * 100.0, 2), "raw_sigmoid": prob}
