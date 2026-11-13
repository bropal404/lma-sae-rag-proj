"""Paired, question-level bootstrap for router comparisons.

Important: sample *row indices* with replacement. Sampling question IDs and then
turning the sample into a dictionary silently drops duplicated questions and
underestimates or otherwise distorts sampling uncertainty.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..schema import ACTIONS


def _stats(y: np.ndarray, actions: np.ndarray, closed: np.ndarray,
           rag: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    truth, act, c, r = y[indices], actions[indices], closed[indices], rag[indices]
    present = [a for a in ACTIONS if np.any(truth == a) or np.any(act == a)]
    attempted = act != "ABSTAIN"
    correct = np.where(act == "ANSWER", c, np.where(act == "RETRIEVE", r, False))
    return {
        "macro_f1": float(f1_score(truth, act, labels=present, average="macro", zero_division=0)),
        "accuracy": float(np.mean(correct)),  # abstentions count as not correct
        "hallucination": float(np.mean(~correct[attempted])) if attempted.any() else float("nan"),
        "retrieval_calls": float(np.mean(act == "RETRIEVE")),
    }


def paired_bootstrap(labels: list[str], first: list[str], second: list[str],
                     closed_correct: list[bool], rag_correct: list[bool], *,
                     draws: int = 2000, seed: int = 13) -> list[dict]:
    """Compute (first - second) differences and percentile CIs over n rows.

    Repeated sampled rows remain repeated when computing each metric. Returns a
    list with the point difference, CI, and number of finite bootstrap draws.
    """
    n = len(labels)
    if not n or any(len(z) != n for z in (first, second, closed_correct, rag_correct)):
        raise ValueError("all inputs must be nonempty and have the same length")
    if draws < 1:
        raise ValueError("draws must be positive")
    y = np.asarray(labels, dtype=object)
    a = np.asarray(first, dtype=object)
    b = np.asarray(second, dtype=object)
    c = np.asarray(closed_correct, dtype=bool)
    r = np.asarray(rag_correct, dtype=bool)
    indices = np.arange(n)
    m1, m2 = _stats(y, a, c, r, indices), _stats(y, b, c, r, indices)
    rng = np.random.default_rng(seed)
    sampling = {k: [] for k in m1}
    for _ in range(draws):
        s = rng.integers(0, n, size=n)
        p, q = _stats(y, a, c, r, s), _stats(y, b, c, r, s)
        for k in sampling:
            diff = p[k] - q[k]
            if np.isfinite(diff):
                sampling[k].append(diff)
    out = []
    for metric, vals in sampling.items():
        lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
        out.append({"metric": metric, "sae_minus_dense": m1[metric] - m2[metric],
                    "ci_low": float(lo), "ci_high": float(hi), "bootstrap_draws": len(vals),
                    "significant": bool(np.isfinite(lo) and (hi < 0 or lo > 0))})
    return out
