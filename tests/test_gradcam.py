import numpy as np
import pytest
import torch

from skin_lesion.explain.gradcam import GradCAM, overlay_heatmap
from skin_lesion.models.factory import build_model


@pytest.mark.parametrize("name", ["baseline", "hybrid"])
def test_gradcam_range_and_shape(tiny_cfg, name):
    tiny_cfg["model"]["name"] = name
    model = build_model(tiny_cfg).eval()
    cam = GradCAM(model)
    logits, heat = cam(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 7)
    assert heat.shape[0] == 2 and heat.ndim == 3
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6
    cam.close()


def test_overlay_shape_and_dtype():
    img = np.zeros((50, 60, 3), dtype=np.uint8)
    out = overlay_heatmap(img, np.random.rand(4, 4))
    assert out.shape == img.shape and out.dtype == np.uint8
