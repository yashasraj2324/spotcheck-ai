"""YAML config loading with `base:` inheritance and dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def _merge(base: Config, new: Config) -> Config:
    """Recursively merge `new` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving an optional `base:` file relative to it."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (ints, bools, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def flatten(cfg: Config, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to dotted keys (for MLflow params)."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
