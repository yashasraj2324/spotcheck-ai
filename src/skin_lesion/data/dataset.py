"""PyTorch dataset and dataloader construction."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from skin_lesion.config import Config
from skin_lesion.data.transforms import build_transforms


def data_root(cfg: Config) -> Path:
    """Dataset root: env HAM10000_DIR if set, else `data.root` from config."""
    return Path(os.environ.get("HAM10000_DIR") or cfg["data"]["root"])


class LesionDataset(Dataset):
    """Images listed in a metadata frame with `path` (relative to `root`) and `label`."""

    def __init__(self, df: pd.DataFrame, root: Path, transform: Callable) -> None:
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        with Image.open(self.root / row["path"]) as img:
            return self.transform(img.convert("RGB")), int(row["label"])


def load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    """Read the unified metadata file and return {split: frame}, honouring `data.subset`."""
    meta = pd.read_csv(cfg["data"]["metadata"])
    subset = cfg["data"].get("subset")
    splits = {}
    for name in ("train", "val", "test"):
        part = meta[meta["split"] == name]
        if subset:
            part = part.sample(n=min(subset, len(part)), random_state=cfg["seed"])
        splits[name] = part.reset_index(drop=True)
    return splits


def class_counts(labels: pd.Series, num_classes: int) -> torch.Tensor:
    """Per-class sample counts as a float tensor of length `num_classes`."""
    return torch.bincount(torch.as_tensor(labels.values), minlength=num_classes).float()


def build_loaders(cfg: Config, splits: dict[str, pd.DataFrame]) -> dict[str, DataLoader]:
    """Build train/val/test loaders; the train loader optionally uses a weighted sampler."""
    root = data_root(cfg)
    d = cfg["data"]
    common = {"batch_size": d["batch_size"], "num_workers": d["num_workers"]}
    loaders = {}
    for name, df in splits.items():
        train = name == "train"
        ds = LesionDataset(df, root, build_transforms(cfg, train=train))
        sampler = None
        if train and cfg["imbalance"]["weighted_sampler"]:
            counts = class_counts(df["label"], cfg["model"]["num_classes"]).clamp(min=1)
            weights = (1.0 / counts)[torch.as_tensor(df["label"].values)]
            sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True)
        loaders[name] = DataLoader(
            ds, shuffle=train and sampler is None, sampler=sampler, **common
        )
    return loaders
