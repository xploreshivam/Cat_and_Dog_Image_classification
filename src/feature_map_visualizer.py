"""Live Conv2D Feature Map Visualization"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def extract_conv2d_activations(model: tf.keras.Model, img_tensor: np.ndarray):
    """Extracts activation tensors from all Conv2D layers for visual explainability."""
    conv_outputs = [
        layer.output for layer in model.layers if "conv2d" in layer.name.lower()
    ]
    if not conv_outputs:
        return []
    activation_model = Model(inputs=model.inputs, outputs=conv_outputs)
    return activation_model.predict(img_tensor, verbose=0)
