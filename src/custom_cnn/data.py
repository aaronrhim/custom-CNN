from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras.datasets import cifar10


Array = np.ndarray


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10() -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    """Load CIFAR-10 dataset with shapes:
    - x: (N, 32, 32, 3)
    - y: (N, 1)
    """
    return cifar10.load_data()


def preprocess(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    *,
    val_size: int = 10_000,
    random_state: int = 42,
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array], Tuple[Array, Array]]:
    """Normalize to [0,1], shuffle, and split validation set from end of train.

    Returns: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    x_train, y_train = sk_shuffle(x_train, y_train, random_state=random_state)

    if val_size > 0:
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]
    else:
        x_val = np.empty((0,))
        y_val = np.empty((0,))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

