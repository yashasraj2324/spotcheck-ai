"""Build a model from config."""

from __future__ import annotations

from torch import nn

from skin_lesion.config import Config
from skin_lesion.models.baseline import BaselineModel
from skin_lesion.models.hybrid import HybridModel


def build_model(cfg: Config, pretrained: bool | None = None) -> nn.Module:
    """Create the baseline or hybrid model named by `cfg["model"]["name"]`."""
    m = cfg["model"]
    use_pretrained = m["pretrained"] if pretrained is None else pretrained
    if m["name"] == "baseline":
        return BaselineModel(m["backbone"], m["num_classes"], use_pretrained, m["dropout"])
    if m["name"] == "hybrid":
        return HybridModel(
            m["backbone"],
            m["num_classes"],
            use_pretrained,
            cfg["data"]["img_size"],
            m["embed_dim"],
            m["num_blocks"],
            m["num_heads"],
            m["dropout"],
        )
    raise ValueError(f"Unknown model name: {m['name']!r} (expected 'baseline' or 'hybrid')")
