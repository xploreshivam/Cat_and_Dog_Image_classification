"""CNN Layer Design Module """
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_compact_cnn(input_shape: Tuple[int, int, int] = (160, 160, 3)) -> tf.keras.Model:
    """Builds the 4-block Conv2D + BatchNormalization + MaxPooling2D binary classifier."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="pixel_normalization")(inputs)
    for filters in [32, 64, 128, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="Compact_CNN_Cat_Dog")
