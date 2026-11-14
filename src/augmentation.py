"""Image Augmentation Module """
import tensorflow as tf
from tensorflow.keras import layers


def get_training_augmentation() -> tf.keras.Sequential:
    """Returns in-graph geometric and photometric augmentation active only during training."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.10),
        ],
        name="training_only_augmentation",
    )
