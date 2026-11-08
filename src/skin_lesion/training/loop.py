"""One-epoch train and evaluate steps."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from skin_lesion.training.metrics import compute_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Train for one epoch and return the mean loss."""
    model.train()
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            loss = criterion(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        seen += len(y)
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict:
    """Return loss plus classification metrics on `loader`."""
    model.eval()
    total, seen = 0.0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * len(y)
        seen += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    metrics = compute_metrics(np.concatenate(labels), np.concatenate(preds))
    metrics["loss"] = total / max(seen, 1)
    return metrics
