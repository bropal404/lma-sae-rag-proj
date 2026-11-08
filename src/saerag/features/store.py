"""Where features live on disk, shared by extraction and probes.

    data/features/{tag}/qids_{file}.json                 row order for every array below
    data/features/{tag}/L{layer}/resid_last_{file}.npy   [n, d_model] float32
    data/features/{tag}/L{layer}/resid_mean_{file}.npy   mean over the question's tokens
    data/features/{tag}/L{layer}/sae{W}_last_{file}.npz  scipy CSR [n, W] float32
    data/features/{tag}/L{layer}/sae{W}_mean_{file}.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..config import path


def feature_dir(cfg: dict) -> Path:
    return path(cfg, "features", cfg["model"]["tag"])


def save(cfg: dict, file: str, layer: int, kind: str, X) -> Path:
    d = feature_dir(cfg) / f"L{layer}"
    d.mkdir(parents=True, exist_ok=True)
    if sp.issparse(X):
        p = d / f"{kind}_{file}.npz"
        sp.save_npz(p, X.tocsr())
    else:
        p = d / f"{kind}_{file}.npy"
        np.save(p, X)
    return p


def save_qids(cfg: dict, file: str, qids: list[str]) -> None:
    feature_dir(cfg).mkdir(parents=True, exist_ok=True)
    (feature_dir(cfg) / f"qids_{file}.json").write_text(json.dumps(qids))


def load(cfg: dict, file: str, layer: int, kind: str):
    """Returns (X, qids). X is a dense float32 array or a CSR matrix."""
    qids = json.loads((feature_dir(cfg) / f"qids_{file}.json").read_text())
    d = feature_dir(cfg) / f"L{layer}"
    if (p := d / f"{kind}_{file}.npz").exists():
        X = sp.load_npz(p).astype(np.float32)
    else:
        X = np.load(d / f"{kind}_{file}.npy").astype(np.float32)
    assert X.shape[0] == len(qids), f"{kind}_{file}: {X.shape[0]} rows, {len(qids)} qids (--limit run?)"
    return X, qids
