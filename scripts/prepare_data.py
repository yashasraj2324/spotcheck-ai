"""Validate local HAM10000 files, build metadata with a lesion-ID split, write a data report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from skin_lesion.config import load_config
from skin_lesion.data.metadata import build_metadata
from skin_lesion.data.report import build_report
from skin_lesion.data.split import split_by_lesion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--root", help="HAM10000 folder (default: env HAM10000_DIR or data.root)")
    parser.add_argument("--report", default="docs/data_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root or os.environ.get("HAM10000_DIR") or cfg["data"]["root"])
    s = cfg["split"]

    df = split_by_lesion(build_metadata(root), s["seed"], s["val_frac"], s["test_frac"])
    out = Path(cfg["data"]["metadata"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report = build_report(df, s["seed"])
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report)
    print(report)
    print(f"Wrote {out} and {args.report}")


if __name__ == "__main__":
    main()
