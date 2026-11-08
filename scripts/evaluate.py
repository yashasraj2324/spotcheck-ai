"""Evaluate a saved checkpoint on the test split.

Example: python scripts/evaluate.py --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json

import torch

from skin_lesion.config import apply_overrides
from skin_lesion.data.dataset import build_loaders, load_splits
from skin_lesion.models.factory import build_model
from skin_lesion.training.loop import evaluate
from skin_lesion.training.losses import build_criterion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = apply_overrides(ckpt["config"], args.overrides)
    splits = load_splits(cfg)
    loaders = build_loaders(cfg, splits)
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    criterion = build_criterion(cfg, splits["train"], device)
    metrics = evaluate(model, loaders["test"], criterion, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
