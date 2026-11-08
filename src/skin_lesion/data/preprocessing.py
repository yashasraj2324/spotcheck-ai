"""Dermoscopy preprocessing: hair removal (DullRazor-style) and CLAHE colour normalisation."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def remove_hair(
    image: np.ndarray, kernel_size: int = 17, threshold: int = 10, radius: int = 1
) -> np.ndarray:
    """Inpaint dark hair strands in an RGB uint8 image via a black-hat filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def normalize_color(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel of an RGB uint8 image in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    merged = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


class Preprocess:
    """Config-driven preprocessing step usable inside a torchvision Compose (PIL in, PIL out)."""

    def __init__(self, cfg: dict) -> None:
        self.hair = cfg["hair_removal"]
        self.hair_kernel = cfg["hair_kernel"]
        self.clahe = cfg["clahe"]
        self.clahe_clip = cfg["clahe_clip"]
        self.clahe_grid = cfg["clahe_grid"]

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"))
        if self.hair:
            arr = remove_hair(arr, self.hair_kernel)
        if self.clahe:
            arr = normalize_color(arr, self.clahe_clip, self.clahe_grid)
        return Image.fromarray(arr)
