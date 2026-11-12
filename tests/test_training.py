import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from skin_lesion.data.constants import CLASSES
from skin_lesion.training import checkpointing
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


def test_end_to_end_training_on_synthetic_data(synthetic_cfg, tmp_path):
    result = run_training(synthetic_cfg)
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "ck" / "best.pth").is_file()
    assert result["run_id"]

    # per-epoch checkpoint holds everything needed to resume
    state = checkpointing.load_epoch_checkpoint(
        checkpointing.checkpoint_path(synthetic_cfg), torch.device("cpu")
    )
    expected = {"epoch", "model", "optimizer", "scheduler", "best_score", "best_state", "run_id"}
    assert expected <= set(state)
    assert state["epoch"] == 0 and state["run_id"] == result["run_id"]
    assert state["completed"] is True  # run finished, so this checkpoint must not be resumed
    synthetic_cfg["train"]["resume"] = True
    with pytest.raises(RuntimeError, match="completed run"):
        checkpointing.find_resume_state(synthetic_cfg, torch.device("cpu"))

    # the packaged model carries a signature (from the logged input example)
    info = mlflow.models.get_model_info(f"runs:/{result['run_id']}/model")
    assert info.signature is not None
