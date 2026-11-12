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
from skin_lesion.training.checkpointing import (  # noqa: E402
    checkpoint_path,
    find_resume_state,
    mark_completed,
    save_epoch_checkpoint,
    upload_checkpoint,
)
from skin_lesion.training.loop import evaluate, train_epoch  # noqa: E402
from skin_lesion.training.losses import build_criterion  # noqa: E402
from skin_lesion.training.tracking import (  # noqa: E402
    log_pytorch_model,
    log_run_params,
    setup_mlflow,
)


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

    # No LR scheduler is used in this loop, so none is checkpointed (the slot stays for future use).
    scheduler = None
    best_score, best_state, stale, start_epoch = -1.0, None, 0, 0
    resume = find_resume_state(cfg, device)
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scaler is not None and resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        if scheduler is not None and resume.get("scheduler"):
            scheduler.load_state_dict(resume["scheduler"])
        best_score, best_state, stale = resume["best_score"], resume["best_state"], resume["stale"]
        start_epoch = resume["epoch"] + 1
        print(f"resuming MLflow run {resume['run_id']} from epoch {start_epoch + 1}")

    setup_mlflow(cfg)
    run_ctx = (
        mlflow.start_run(run_id=resume["run_id"])
        if resume
        else mlflow.start_run(run_name=cfg["run_name"])
    )
    with run_ctx:
        if not resume:  # a resumed run already has its params; re-logging could conflict
            log_run_params(cfg)
            mlflow.log_params(
                {"device": str(device), **{f"n_{k}": len(v) for k, v in splits.items()}}
            )
        # A resumed run that had already hit the early-stopping limit has no epochs left.
        patience_hit = stale >= t["early_stopping_patience"]
        epochs_to_run = range(0) if patience_hit else range(start_epoch, t["epochs"])
        for epoch in epochs_to_run:
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

            ckpt_path = checkpoint_path(cfg)
            save_epoch_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "best_score": best_score,
                    "best_state": best_state,
                    "stale": stale,
                    "run_id": mlflow.active_run().info.run_id,
                    "config": cfg,
                    "completed": False,
                },
            )
            upload_checkpoint(cfg, ckpt_path)

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
            log_pytorch_model(model, cfg["data"]["img_size"])
        except Exception as exc:  # tracking backends differ in model-logging support
            print(f"log_model failed ({exc}); the checkpoint artifact was logged instead")
        run_id = mlflow.active_run().info.run_id
        mark_completed(cfg)  # only after everything is logged, so a crash earlier stays resumable

    return {"run_id": run_id, "best_val_macro_f1": best_score, "test": test}
