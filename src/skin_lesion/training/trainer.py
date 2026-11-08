"""Full training run: fit with early stopping, then one test evaluation, all logged to MLflow."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from skin_lesion.config import Config  # noqa: E402
from skin_lesion.data.constants import CLASSES  # noqa: E402
from skin_lesion.data.dataset import build_loaders, load_splits  # noqa: E402
from skin_lesion.models.factory import build_model  # noqa: E402
from skin_lesion.training.loop import evaluate, train_epoch  # noqa: E402
from skin_lesion.training.losses import build_criterion  # noqa: E402
from skin_lesion.training.tracking import log_run_params, setup_mlflow  # noqa: E402


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.backbone_parameters():
        p.requires_grad = trainable


def _confusion_figure(cm: list[list[int]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def save_checkpoint(model: torch.nn.Module, cfg: Config, path: Path) -> None:
    """Save weights together with the config needed to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": CLASSES}, path)


def run_training(cfg: Config) -> dict:
    """Train, pick the best epoch by `train.monitor`, evaluate on test once, log everything."""
    seed_everything(cfg["seed"])
    t = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(cfg)
    loaders = build_loaders(cfg, splits)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, splits["train"], device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": t["backbone_lr"]},
            {"params": model.head_parameters(), "lr": t["head_lr"]},
        ],
        weight_decay=t["weight_decay"],
    )
    scaler = torch.amp.GradScaler() if (t["amp"] and device.type == "cuda") else None

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=cfg["run_name"]):
        log_run_params(cfg)
        mlflow.log_params(
            {"device": str(device), **{f"n_{k}": len(v) for k, v in splits.items()}}
        )
        best_score, best_state, stale = -1.0, None, 0
        for epoch in range(t["epochs"]):
            _set_backbone_trainable(model, epoch >= t["warmup_epochs"])
            train_loss = train_epoch(model, loaders["train"], criterion, optimizer, device, scaler)
            val = evaluate(model, loaders["val"], criterion, device)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_macro_f1": val["macro_f1"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch + 1}/{t['epochs']} train_loss={train_loss:.4f} "
                f"val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}"
            )
            score = val["macro_f1"]
            if score > best_score:
                best_score, stale = score, 0
                best_state = copy.deepcopy(model.state_dict())
                mlflow.log_metric("best_val_macro_f1", best_score, step=epoch)
            else:
                stale += 1
                if stale >= t["early_stopping_patience"]:
                    print(f"early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_state)
        test = evaluate(model, loaders["test"], criterion, device)
        mlflow.log_metrics(
            {
                "test_macro_f1": test["macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_melanoma_recall": test["melanoma_recall"],
                **{f"test_recall_{c}": r for c, r in test["per_class_recall"].items()},
            }
        )
        mlflow.log_figure(_confusion_figure(test["confusion_matrix"]), "test_confusion_matrix.png")
        mlflow.log_dict(test, "test_metrics.json")

        ckpt = Path(t["checkpoint_dir"]) / "best.pth"
        save_checkpoint(model, cfg, ckpt)
        mlflow.log_artifact(str(ckpt), artifact_path="checkpoint")
        try:
            mlflow.pytorch.log_model(model.cpu(), name="model")
        except Exception as exc:  # tracking backends differ in model-logging support
            print(f"log_model failed ({exc}); the checkpoint artifact was logged instead")
        run_id = mlflow.active_run().info.run_id

    return {"run_id": run_id, "best_val_macro_f1": best_score, "test": test}
