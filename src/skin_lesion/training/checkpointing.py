"""Per-epoch training checkpoints (for resume), with optional upload to Azure Blob Storage.

The Blob upload is enabled only when AZURE_STORAGE_CONNECTION_STRING is set in the environment;
the connection string is never read from config or hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from skin_lesion.config import Config, flatten

CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
LAST_CHECKPOINT = "last.pth"
# Config keys allowed to differ between the saved run and the resuming run.
RESUME_IGNORED_KEYS = frozenset({"train.resume"})
_MISSING = object()


def checkpoint_path(cfg: Config) -> Path:
    """Where the rolling per-epoch checkpoint lives."""
    return Path(cfg["train"]["checkpoint_dir"]) / LAST_CHECKPOINT


def blob_name(cfg: Config) -> str:
    """Blob path in the container, namespaced by run name so runs do not overwrite each other."""
    return f"{cfg['run_name']}/{LAST_CHECKPOINT}"


def save_epoch_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write `state` atomically (temp file + rename) so a crash mid-save keeps the last good one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_epoch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint written by `save_epoch_checkpoint` (trusted, locally produced file)."""
    return torch.load(path, map_location=device, weights_only=False)


def _container_client(container: str):
    """Blob container client from the env connection string; creates the container if missing."""
    from azure.storage.blob import BlobServiceClient  # lazy: only needed with the azure extra

    service = BlobServiceClient.from_connection_string(os.environ[CONNECTION_STRING_ENV])
    client = service.get_container_client(container)
    if not client.exists():
        client.create_container()
    return client


def upload_checkpoint(cfg: Config, path: Path) -> bool:
    """Upload `path` to the checkpoints container if configured. Never raises: returns success."""
    if not os.environ.get(CONNECTION_STRING_ENV):
        return False
    try:
        client = _container_client(cfg["train"]["checkpoint_container"])
        with path.open("rb") as f:
            client.upload_blob(blob_name(cfg), f, overwrite=True)
        return True
    except Exception as exc:  # a failed upload must not kill a long training run
        print(f"warning: checkpoint upload to Blob failed ({type(exc).__name__}: {exc})")
        return False


def download_checkpoint(cfg: Config, path: Path) -> bool:
    """Fetch the checkpoint from Blob into `path` if configured and present. Never raises."""
    if not os.environ.get(CONNECTION_STRING_ENV):
        return False
    try:
        client = _container_client(cfg["train"]["checkpoint_container"])
        blob = client.get_blob_client(blob_name(cfg))
        if not blob.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob.download_blob().readall())
        return True
    except Exception as exc:
        print(f"warning: checkpoint download from Blob failed ({type(exc).__name__}: {exc})")
        return False


def config_differences(saved: Config | None, current: Config) -> list[str]:
    """Dotted config keys whose values differ (ignoring `RESUME_IGNORED_KEYS`)."""
    old, new = flatten(saved or {}), flatten(current)
    keys = (old.keys() | new.keys()) - RESUME_IGNORED_KEYS
    return sorted(k for k in keys if old.get(k, _MISSING) != new.get(k, _MISSING))


def validate_resume_state(state: dict[str, Any], cfg: Config) -> None:
    """Refuse to resume a finished run or a run trained with a different config."""
    if state.get("completed"):
        raise RuntimeError(
            "The checkpoint belongs to a completed run and cannot be resumed. "
            "Start a new run (set train.resume=false) or delete the checkpoint."
        )
    diff = config_differences(state.get("config"), cfg)
    if diff:
        shown = ", ".join(diff[:10]) + (" ..." if len(diff) > 10 else "")
        raise ValueError(
            f"Refusing to resume: the checkpoint's saved config differs from the current config "
            f"in {len(diff)} key(s): {shown}. Restore the original config or start a new run."
        )


def find_resume_state(cfg: Config, device: torch.device) -> dict[str, Any] | None:
    """Return the saved training state when `train.resume` is true and a checkpoint exists.

    Falls back to downloading from Blob when there is no local file (fresh Colab/Kaggle disk).
    Raises if the checkpoint is from a completed run or was saved with a different config.
    """
    if not cfg["train"].get("resume"):
        return None
    path = checkpoint_path(cfg)
    if not path.is_file() and not download_checkpoint(cfg, path):
        print("resume requested but no checkpoint found; starting a fresh run")
        return None
    state = load_epoch_checkpoint(path, device)
    validate_resume_state(state, cfg)
    return state


def mark_completed(cfg: Config) -> None:
    """Flag the rolling checkpoint (local and Blob copy) as belonging to a finished run."""
    path = checkpoint_path(cfg)
    if not path.is_file():
        return
    state = load_epoch_checkpoint(path, torch.device("cpu"))
    state["completed"] = True
    save_epoch_checkpoint(path, state)
    upload_checkpoint(cfg, path)
