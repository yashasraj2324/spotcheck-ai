"""Grad-CAM for any model that exposes a `cam_layer` module producing a (B, C, H, W) feature map."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM via a forward hook plus a tensor hook on the layer output.

    A tensor hook (rather than `register_full_backward_hook`) is used because timm's
    in-place activations make module backward hooks unreliable.
    """

    def __init__(self, model: nn.Module, layer: nn.Module | None = None) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = (layer or model.cam_layer).register_forward_hook(self._on_forward)

    def _on_forward(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self.activations = out
        if out.requires_grad:
            out.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return (logits, cam) where cam is `(B, H, W)` in [0, 1] for the target class."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            target = logits.argmax(1) if class_idx is None else torch.full((len(x),), class_idx)
            logits[torch.arange(len(x)), target].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return logits.detach(), cam.detach().cpu().numpy()

    def close(self) -> None:
        """Remove the forward hook."""
        self._handle.remove()


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap (any size) onto an RGB uint8 image; returns RGB uint8."""
    heat = cv2.resize(cam.astype(np.float32), (image.shape[1], image.shape[0]))
    jet = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    color = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    blend = image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.uint8(np.clip(blend, 0, 255))
