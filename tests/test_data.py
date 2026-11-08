import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from skin_lesion.data.constants import CLASSES
from skin_lesion.data.dataset import build_loaders
from skin_lesion.data.preprocessing import Preprocess, normalize_color, remove_hair
from skin_lesion.data.split import assert_no_lesion_leakage, split_by_lesion
from skin_lesion.data.transforms import build_transforms


def _rgb(size=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _metadata(n_lesions=200, seed=0):
    """Synthetic HAM10000-like table: 1-3 images per lesion, one dx per lesion."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_lesions):
        dx = CLASSES[i % len(CLASSES)]
        for j in range(rng.integers(1, 4)):
            rows.append({"lesion_id": f"L{i}", "image_id": f"L{i}_{j}", "dx": dx})
    return pd.DataFrame(rows)


def test_preprocessing_keeps_shape_and_dtype():
    img = _rgb()
    assert remove_hair(img).shape == img.shape
    out = normalize_color(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_preprocess_step_pil_roundtrip(tiny_cfg):
    out = Preprocess(tiny_cfg["preprocess"])(Image.fromarray(_rgb()))
    assert out.size == (64, 64) and out.mode == "RGB"


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output(tiny_cfg, train):
    tensor = build_transforms(tiny_cfg, train=train)(Image.fromarray(_rgb(100)))
    assert tensor.shape == (3, 64, 64) and tensor.dtype == torch.float32


def test_split_has_no_lesion_leakage():
    out = split_by_lesion(_metadata(), seed=42, val_frac=0.15, test_frac=0.15)
    assert set(out["split"]) == {"train", "val", "test"}
    assert_no_lesion_leakage(out)
    sets = {s: set(g["lesion_id"]) for s, g in out.groupby("split")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_split_is_deterministic_and_seed_sensitive():
    df = _metadata()
    a = split_by_lesion(df, 42, 0.15, 0.15)
    b = split_by_lesion(df, 42, 0.15, 0.15)
    c = split_by_lesion(df, 7, 0.15, 0.15)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != c["split"].tolist()


def test_leakage_check_catches_leaks():
    df = split_by_lesion(_metadata(), 42, 0.15, 0.15)
    multi = df.groupby("lesion_id").size()
    lesion = multi[multi > 1].index[0]
    idx = df.index[df["lesion_id"] == lesion]
    df.loc[idx[0], "split"] = "train"
    df.loc[idx[1], "split"] = "test"
    with pytest.raises(AssertionError):
        assert_no_lesion_leakage(df)


def test_weighted_sampler_balances_classes(tiny_cfg, tmp_path):
    labels = [0] * 90 + [1] * 10
    rows = []
    for i, label in enumerate(labels):
        Image.fromarray(_rgb()).save(tmp_path / f"{i}.jpg")
        rows.append({"path": f"{i}.jpg", "label": label})
    tiny_cfg["data"].update(root=str(tmp_path), batch_size=100, num_workers=0)
    tiny_cfg["imbalance"]["weighted_sampler"] = True
    loaders = build_loaders(tiny_cfg, {"train": pd.DataFrame(rows)})
    drawn = np.concatenate([y.numpy() for _, y in loaders["train"]])
    assert 0.3 < drawn.mean() < 0.7  # ~50% minority vs 10% unweighted
