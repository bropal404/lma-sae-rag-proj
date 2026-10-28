"""Config loading. Every script takes --config (default configs/default.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
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
