import pandas as pd
import pytest

from skin_lesion.data.constants import CLASSES
from skin_lesion.data.report import build_report
from skin_lesion.data.split import split_by_lesion


def _df():
    rows = [
        {"lesion_id": f"L{i}", "image_id": f"L{i}_{j}", "dx": CLASSES[i % 7]}
        for i in range(140)
        for j in range(2)
    ]
    return pd.DataFrame(rows)


def test_report_lists_every_class_and_passes_leak_check():
    report = build_report(split_by_lesion(_df(), 42, 0.15, 0.15), 42)
    assert "PASSED" in report
    assert all(f"| {c} |" in report for c in CLASSES)


def test_report_refuses_leaky_split():
    df = split_by_lesion(_df(), 42, 0.15, 0.15)
    rows = df[df["lesion_id"] == "L0"].index
    df.loc[rows[0], "split"] = "train"
    df.loc[rows[1], "split"] = "test"
    with pytest.raises(AssertionError):
        build_report(df, 42)
