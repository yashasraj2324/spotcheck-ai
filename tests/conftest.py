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


@pytest.fixture
def synthetic_cfg(tiny_cfg, tmp_path, monkeypatch):
    """Tiny config using synthetic images, a tmp SQLite MLflow store and tmp checkpoints."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(42):
        arr = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"{i}.jpg")
        split = ("train", "val", "test")[i % 3]
        rows.append({"path": f"{i}.jpg", "label": i % 7, "split": split})
    pd.DataFrame(rows).to_csv(tmp_path / "meta.csv", index=False)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("HAM10000_DIR", str(tmp_path))
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    tiny_cfg["data"].update(metadata=str(tmp_path / "meta.csv"), batch_size=8, num_workers=0)
    tiny_cfg["train"].update(
        epochs=1, warmup_epochs=0, amp=False, checkpoint_dir=str(tmp_path / "ck")
    )
    tiny_cfg["preprocess"]["hair_removal"] = False
    return tiny_cfg
