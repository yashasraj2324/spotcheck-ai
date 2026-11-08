"""Evaluation metrics for multi-class lesion classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from skin_lesion.data.constants import CLASSES, MELANOMA_IDX


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-F1, accuracy, per-class recall (by class name) and melanoma recall."""
    labels = list(range(len(CLASSES)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, recalls, strict=True)},
        "melanoma_recall": float(recalls[MELANOMA_IDX]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
