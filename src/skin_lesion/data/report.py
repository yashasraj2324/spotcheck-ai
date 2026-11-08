"""Data report: class counts per split plus a lesion-leakage confirmation."""

from __future__ import annotations

import pandas as pd

from skin_lesion.data.constants import CLASSES
from skin_lesion.data.split import assert_no_lesion_leakage

SPLITS = ("train", "val", "test")


def build_report(df: pd.DataFrame, seed: int) -> str:
    """Return a markdown report. Raises AssertionError if any lesion_id spans multiple splits."""
    assert_no_lesion_leakage(df)
    counts = pd.crosstab(df["dx"], df["split"]).reindex(index=CLASSES, columns=list(SPLITS), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum()
    lesions = df.groupby("split")["lesion_id"].nunique().reindex(list(SPLITS))
    lines = [
        "# HAM10000 data report",
        "",
        f"- Images: {len(df)}, lesions: {df['lesion_id'].nunique()}, split seed: {seed}",
        "- Split unit: `lesion_id` (stratified by diagnosis)",
        "- Leakage check: PASSED, no lesion_id appears in more than one split",
        f"- Lesions per split: " + ", ".join(f"{s}={lesions[s]}" for s in SPLITS),
        "",
        "## Images per class and split",
        "",
        "| class | " + " | ".join(counts.columns) + " |",
        "|---|" + "---|" * len(counts.columns),
    ]
    for name, row in counts.iterrows():
        lines.append(f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(lines) + "\n"
