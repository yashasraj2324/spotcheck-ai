import numpy as np
import pandas as pd
import torch
from PIL import Image

from skin_lesion.data.constants import CLASSES
from skin_lesion.training.losses import build_criterion
from skin_lesion.training.metrics import compute_metrics
from skin_lesion.training.trainer import run_training


def test_metrics_perfect_and_melanoma_recall():
    y = np.arange(len(CLASSES))
    m = compute_metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["melanoma_recall"] == 1.0
    wrong = y.copy()
    wrong[CLASSES.index("mel")] = 0
    assert compute_metrics(y, wrong)["melanoma_recall"] == 0.0


def test_class_weighted_loss_weights_rare_classes_more(tiny_cfg):
    tiny_cfg["imbalance"]["class_weighted_loss"] = True
    df = pd.DataFrame({"label": [0] * 50 + [1] * 5 + list(range(2, 7))})
    crit = build_criterion(tiny_cfg, df, torch.device("cpu"))
    assert crit.weight[1] > crit.weight[0]


def test_no_weights_by_default(tiny_cfg):
    crit = build_criterion(tiny_cfg, pd.DataFrame({"label": [0, 1]}), torch.device("cpu"))
    assert crit.weight is None


def test_end_to_end_training_on_synthetic_data(tiny_cfg, tmp_path, monkeypatch):
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
    tiny_cfg["data"].update(metadata=str(tmp_path / "meta.csv"), batch_size=8, num_workers=0)
    tiny_cfg["train"].update(
        epochs=1, warmup_epochs=0, amp=False, checkpoint_dir=str(tmp_path / "ck")
    )
    tiny_cfg["preprocess"]["hair_removal"] = False

    result = run_training(tiny_cfg)
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "ck" / "best.pth").is_file()
    assert result["run_id"]
