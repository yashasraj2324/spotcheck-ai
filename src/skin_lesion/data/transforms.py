"""Train/eval transform pipelines. Preprocessing runs after resize, identically in training and serving."""

from __future__ import annotations

import torchvision.transforms as T

from skin_lesion.config import Config
from skin_lesion.data.constants import IMAGENET_MEAN, IMAGENET_STD
from skin_lesion.data.preprocessing import Preprocess


def build_transforms(cfg: Config, train: bool) -> T.Compose:
    """Return a PIL -> normalised tensor pipeline; augmentation only when `train`."""
    size = cfg["data"]["img_size"]
    steps: list = [T.Resize((size, size)), Preprocess(cfg["preprocess"])]
    if train:
        a = cfg["augment"]
        if a["hflip"]:
            steps.append(T.RandomHorizontalFlip())
        if a["vflip"]:
            steps.append(T.RandomVerticalFlip())
        if a["rotation"]:
            steps.append(T.RandomRotation(a["rotation"]))
        steps.append(
            T.ColorJitter(
                brightness=a["brightness"], contrast=a["contrast"], saturation=a["saturation"]
            )
        )
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)
