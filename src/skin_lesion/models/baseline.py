"""Plain EfficientNet baseline."""

from __future__ import annotations

import timm
import torch
from torch import nn


class BaselineModel(nn.Module):
    """timm EfficientNet with a fresh classifier head."""

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits `(B, num_classes)`."""
        return self.net(x)

    @property
    def cam_layer(self) -> nn.Module:
        """Layer whose output feature map Grad-CAM uses."""
        return self.net.blocks[-1]

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of everything except the classifier."""
        head = {id(p) for p in self.net.get_classifier().parameters()}
        return [p for p in self.parameters() if id(p) not in head]

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the classifier head."""
        return list(self.net.get_classifier().parameters())
