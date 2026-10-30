"""Feature analysis for one SAE probe. Owner: Person 3.

    uv run python -m saerag.probes.analyze --train nq_train triviaqa_train --eval nq_test \
        --labels default --layer 17 --kind sae16k_last

Fits the probe on --train (or a 70/30 split of it when --eval is not given), then writes to
runs/probes/{tag}/{labels}/analysis__{kind}@L{layer}/:
    top_features.csv    top features per class: weight, firing rate in and out of the class
    hand_check.csv      30 examples per class by default, with their strongest features and
                        an empty `verdict` column to fill in by hand (Neuronpedia for meanings)
    drop_top_k.csv      macro-F1 after removing the top-k features of every class, measured
                        two ways: zeroed at test time (how much the fitted probe leans on them)
                        and retrained without them (whether others can replace them)
Features are ranked on training data only, so the drop test does not peek at the eval set.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from ..config import load_config, path
from .train import load_block, make_model, score

DROP_K = [0, 1, 5, 10, 25, 50, 100]


def rank_features(coef: np.ndarray) -> np.ndarray:
    """[n_classes, W] weights to [n_classes, W] feature ids, strongest evidence for the class first."""
    return np.argsort(-coef, axis=1)


def top_union(ranked: np.ndarray, k: int) -> np.ndarray:
    return np.unique(ranked[:, :k]) if k else np.array([], dtype=int)


def zero_columns(X: sp.csr_matrix, cols: np.ndarray) -> sp.csr_matrix:
    keep = np.ones(X.shape[1])
    keep[cols] = 0
    return (X @ sp.diags(keep)).tocsr()


def top_table(X: sp.csr_matrix, y: np.ndarray, classes: list[str], coef: np.ndarray,
              ranked: np.ndarray, k: int) -> pd.DataFrame:
    fires = X > 0
    rows = []
    for j, c in enumerate(classes):
        mask = y == j
        for rank, f in enumerate(ranked[j, :k], 1):
            hits = np.asarray(fires[:, f].todense()).ravel()
            rows.append({"class": c, "rank": rank, "feature": int(f), "weight": float(coef[j, f]),
                         "rate_in_class": float(hits[mask].mean()),
                         "rate_elsewhere": float(hits[~mask].mean())})
    return pd.DataFrame(rows)


def hand_check_sheet(X: sp.csr_matrix, y: np.ndarray, p: np.ndarray, qids: list[str],
                     texts: np.ndarray, classes: list[str], coef: np.ndarray, n: int,
                     seed: int) -> pd.DataFrame:
    """A random sample per predicted class, so the check covers routine decisions too."""
    pred = p.argmax(1)
    rng = np.random.default_rng(seed)
    rows = []
    for j, c in enumerate(classes):
        idx = np.flatnonzero(pred == j)
        for i in rng.choice(idx, min(n, len(idx)), replace=False):
            row = X[i].tocoo()
            evidence = row.data * coef[j, row.col]
            top = row.col[np.argsort(-evidence)[:5]]
            rows.append({"qid": qids[i], "question": texts[i], "predicted": c, "label": classes[y[i]],
                         "confidence": round(float(p[i, j]), 3),
                         "top_features": " ".join(map(str, top)), "verdict": ""})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--labels", default="default")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--kind", default="sae16k_last")
    ap.add_argument("--penalty", choices=["l1", "l2"], default="l2")
    ap.add_argument("--top", type=int, default=20, help="features listed per class")
    ap.add_argument("--examples", type=int, default=30, help="hand-check examples per class")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    name = f"{args.kind}@L{args.layer}"

    qids, y_str, texts, feats = load_block(cfg, args.train, args.labels, [args.layer], [args.kind])
    X = feats[name]
    classes = sorted(set(y_str))
    y = np.array([classes.index(c) for c in y_str])
    if args.eval:
        q2, y2, t2, f2 = load_block(cfg, args.eval, args.labels, [args.layer], [args.kind])
        Xtr, ytr, Xte, qte, tte = X, y, f2[name], q2, t2
        yte = np.array([classes.index(c) for c in y2])
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        qte, tte = [qids[i] for i in te], texts[te]

    search = make_model(name, args.penalty, False, seed).fit(Xtr, ytr)
    fitted = search.best_estimator_
    coef = fitted[-1].coef_
    coef = np.vstack([-coef[0], coef[0]]) if coef.shape[0] == 1 else coef
    ranked = rank_features(coef)
    p = search.predict_proba(Xte)

    out = path(cfg, "results", "probes", cfg["model"]["tag"], args.labels, f"analysis__{name}")
    out.mkdir(parents=True, exist_ok=True)
    top_table(Xtr, ytr, classes, coef, ranked, args.top).to_csv(out / "top_features.csv", index=False)
    hand_check_sheet(Xte, yte, p, qte, tte, classes, coef, args.examples, seed).to_csv(
        out / "hand_check.csv", index=False)

    rows = []
    for k in DROP_K:
        cols = top_union(ranked, k)
        zeroed = score(fitted.predict_proba(zero_columns(Xte, cols)), yte, len(classes))
        retrain = clone(fitted).fit(zero_columns(Xtr, cols), ytr)
        refit = score(retrain.predict_proba(zero_columns(Xte, cols)), yte, len(classes))
        rows.append({"k": k, "features_removed": len(cols), "f1_zeroed": zeroed["macro_f1"],
                     "f1_retrained": refit["macro_f1"], "auroc_zeroed": zeroed["auroc"],
                     "auroc_retrained": refit["auroc"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "drop_top_k.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
