"""Shared test fixtures: a tiny config so tests never touch the real dataset."""

import pytest

from skin_lesion.config import load_config

TINY_IMG = 64


@pytest.fixture
def tiny_cfg():
    """Baseline config shrunk to 64px inputs, no pretrained weights."""
    cfg = load_config("configs/baseline.yaml")
    cfg["data"]["img_size"] = TINY_IMG
    cfg["model"]["pretrained"] = False
    return cfg
