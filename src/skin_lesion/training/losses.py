"""Loss construction with optional class weighting."""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from skin_lesion.config import Config
from skin_lesion.data.dataset import class_counts


def build_criterion(cfg: Config, train_df: pd.DataFrame, device: torch.device) -> nn.Module:
    """Cross-entropy; inverse-frequency class weights when `imbalance.class_weighted_loss`."""
    if not cfg["imbalance"]["class_weighted_loss"]:
        return nn.CrossEntropyLoss()
    k = cfg["model"]["num_classes"]
    counts = class_counts(train_df["label"], k).clamp(min=1)
    weights = counts.sum() / (k * counts)
    return nn.CrossEntropyLoss(weight=weights.to(device))
