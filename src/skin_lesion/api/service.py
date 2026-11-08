"""Inference service: preprocessing, class probabilities and Grad-CAM overlay."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from skin_lesion.config import Config
from skin_lesion.data.constants import CLASSES
from skin_lesion.data.transforms import build_transforms
from skin_lesion.explain.gradcam import GradCAM, overlay_heatmap
from skin_lesion.models.factory import build_model


class Predictor:
    """Wraps a trained model, its eval transform and Grad-CAM."""

    def __init__(self, model: torch.nn.Module, cfg: Config, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.size = cfg["data"]["img_size"]
        self.transform = build_transforms(cfg, train=False)
        self.cam = GradCAM(self.model)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> Predictor:
        """Rebuild the model from the config stored in the checkpoint and load its weights."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(ckpt["config"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["config"], device)

    def predict(self, image: Image.Image) -> dict:
        """Return probabilities per class, the top class and a base64 PNG Grad-CAM overlay."""
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits, cam = self.cam(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        shown = np.asarray(image.resize((self.size, self.size)))
        buf = io.BytesIO()
        Image.fromarray(overlay_heatmap(shown, cam[0])).save(buf, format="PNG")
        return {
            "probabilities": {c: float(p) for c, p in zip(CLASSES, probs, strict=True)},
            "top_class": CLASSES[top],
            "confidence": float(probs[top]),
            "gradcam_png_base64": base64.b64encode(buf.getvalue()).decode(),
        }
