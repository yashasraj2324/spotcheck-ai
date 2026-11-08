"""Locate HAM10000 files and build the unified metadata table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skin_lesion.data.constants import CLASS_TO_IDX

METADATA_FILE = "HAM10000_metadata.csv"
IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")
REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}


def build_metadata(root: str | Path) -> pd.DataFrame:
    """Validate the raw HAM10000 layout and return image_id, lesion_id, dx, label, path.

    `path` is relative to `root` so the table stays valid wherever the data is mounted.
    Raises FileNotFoundError / ValueError with an actionable message on any problem.
    """
    root = Path(root)
    csv = root / METADATA_FILE
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found")
    missing_dirs = [d for d in IMAGE_DIRS if not (root / d).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing image folders under {root}: {missing_dirs}")

    df = pd.read_csv(csv)
    if not REQUIRED_COLUMNS <= set(df.columns):
        raise ValueError(f"Metadata must have columns {sorted(REQUIRED_COLUMNS)}")
    unknown = set(df["dx"]) - set(CLASS_TO_IDX)
    if unknown:
        raise ValueError(f"Unknown diagnosis labels: {sorted(unknown)}")

    index = {
        p.stem: p.relative_to(root).as_posix()
        for d in IMAGE_DIRS
        for p in (root / d).glob("*.jpg")
    }
    df["path"] = df["image_id"].map(index)
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        raise FileNotFoundError(f"{n_missing} metadata rows have no image file under {root}")
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    return df[["image_id", "lesion_id", "dx", "label", "path"]]
