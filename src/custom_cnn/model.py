from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, int, int] = (32, 32, 3), num_classes: int = 10) -> keras.Model:
    # fully custom sequential model

    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        # convolution without bias and keep at 32x32
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 16x16

        # custom feature extration 
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        layers.Conv2D(256, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Flatten using global average pooling
        layers.GlobalAveragePooling2D(),

        # hyperparameterization with regularization
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4)),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    
    return model

