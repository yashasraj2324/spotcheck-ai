"""Train a model from a YAML config.

Example: python scripts/train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json

from skin_lesion.config import apply_overrides, load_config
from skin_lesion.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.epochs=1 (repeatable)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    result = run_training(cfg)
    print(json.dumps({"run_id": result["run_id"], "test_macro_f1": result["test"]["macro_f1"]}))


if __name__ == "__main__":
    main()
