"""MLflow setup and run provenance. The tracking URI comes from MLFLOW_TRACKING_URI."""

from __future__ import annotations

import os
import subprocess

import mlflow

from skin_lesion.config import Config, flatten

# MLflow >= 3.x disables the plain ./mlruns file store, so fall back to a local SQLite file.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def git_info() -> dict[str, str]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain"))),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def setup_mlflow(cfg: Config) -> None:
    """Point MLflow at MLFLOW_TRACKING_URI (default: local SQLite file) and pick the experiment."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def log_run_params(cfg: Config) -> None:
    """Log every config value plus git commit and dataset asset version."""
    mlflow.log_params({k: str(v) for k, v in flatten(cfg).items()})
    mlflow.log_params(
        {
            **git_info(),
            "dataset_asset_name": os.environ.get("DATASET_ASSET_NAME", "unset"),
            "dataset_asset_version": os.environ.get("DATASET_ASSET_VERSION", "unset"),
        }
    )
