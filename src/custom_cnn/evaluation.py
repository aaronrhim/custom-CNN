from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras


def evaluate(model: keras.Model, x_test, y_test, *, batch_size: int = 128):
    return model.evaluate(x_test, y_test, batch_size=batch_size)


def predict(model: keras.Model, x, *, batch_size: int | None = None):
    return model.predict(x, batch_size=batch_size)


def classification_report_from_predictions(y_true, y_pred_proba):
    """Compute a text classification report from probability outputs."""
    y_classes = np.argmax(y_pred_proba, axis=1)
    # y_true might be shape (N,1); flatten to (N,)
    y_true = np.asarray(y_true).reshape(-1,)
    return classification_report(y_true, y_classes)

