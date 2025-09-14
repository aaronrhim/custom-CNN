from __future__ import annotations

from typing import Sequence

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def compile_model(
    model: keras.Model,
    *,
    optimizer: str | keras.optimizers.Optimizer = "adam",
    loss: str | keras.losses.Loss = "sparse_categorical_crossentropy",
    metrics: Sequence[str] | None = None,
) -> keras.Model:
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))
    return model


def fit(
    model: keras.Model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=reduce_lr_patience, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    return history

