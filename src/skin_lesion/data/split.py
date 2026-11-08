"""Deterministic lesion-ID-based train/val/test split.

HAM10000 has several images per lesion, so splitting by image leaks near-duplicates across
splits. We split on `lesion_id` and stratify by diagnosis (each lesion has a single `dx`).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_lesion(
    df: pd.DataFrame, seed: int, val_frac: float, test_frac: float
) -> pd.DataFrame:
    """Return a copy of `df` with a `split` column in {train, val, test}."""
    lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]
    if lesions["lesion_id"].duplicated().any() or df.groupby("lesion_id")["dx"].nunique().max() > 1:
        raise ValueError("Each lesion_id must map to exactly one diagnosis")
    train_ids, hold_ids = train_test_split(
        lesions, test_size=val_frac + test_frac, stratify=lesions["dx"], random_state=seed
    )
    val_ids, test_ids = train_test_split(
        hold_ids,
        test_size=test_frac / (val_frac + test_frac),
        stratify=hold_ids["dx"],
        random_state=seed,
    )
    lookup = {
        **dict.fromkeys(train_ids["lesion_id"], "train"),
        **dict.fromkeys(val_ids["lesion_id"], "val"),
        **dict.fromkeys(test_ids["lesion_id"], "test"),
    }
    out = df.copy()
    out["split"] = out["lesion_id"].map(lookup)
    assert_no_lesion_leakage(out)
    return out


def assert_no_lesion_leakage(df: pd.DataFrame) -> None:
    """Raise if any lesion_id appears in more than one split or has no split."""
    if df["split"].isna().any():
        raise AssertionError("Some rows have no split assigned")
    n_splits = df.groupby("lesion_id")["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} lesion_ids appear in more than one split")
