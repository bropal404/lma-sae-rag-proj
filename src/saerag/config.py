"""Config loading. Every script takes --config (default configs/default.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # `extends: other.yaml` overlays this file on another config in the same directory.
    if base := cfg.pop("extends", None):
        cfg = _merge(load_config(path.parent / base), cfg)
    # SAERAG_DATA points every data path at a shared directory (e.g. on the GPU cluster).
    data_root = os.environ.get("SAERAG_DATA")
    if data_root:
        old = cfg["paths"]["root"]
        for key, value in cfg["paths"].items():
            if isinstance(value, str) and value.startswith(old):
                cfg["paths"][key] = data_root + value[len(old):]
    return cfg


def path(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    p = Path(cfg["paths"][key], *parts)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p
