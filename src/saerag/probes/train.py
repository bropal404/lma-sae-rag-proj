"""SAE and dense probes (B7) on the same activation, plus floor baselines. Owner: Person 3.

    # MVP: proxy labels from the closed-book run (ANSWER vs NOT_ANSWER), 5-fold CV on the pilot
    uv run python -m saerag.probes.train --train pilot --labels closed_hf --layers 17

    # Real labels: train on NQ+TQA train, test on held-out files
    uv run python -m saerag.probes.train --train nq_train triviaqa_train \
        --eval nq_val triviaqa_val nq_test popqa_test --labels default --layers 9 17 22 29

Every row of the comparison uses the same folds and the same label budget:
    majority     predicts the most common training class
    text-tfidf   question text only (a quick B8; Person 4 owns the full one)
    <kind>@L     logistic regression on features/{tag}/L{layer}/{kind}_{file}
Hyperparameter C is picked by 3-fold inner CV on macro-F1. Outputs:
    runs/probes/{tag}/{labels}/{run}__{penalty}/summary.csv   one row per (model, budget, eval)
    runs/routers/{probe_id}/probs_{file}.jsonl     out-of-fold / test probabilities (full budget)
Pass the val files in --eval too: Person 4 tunes thresholds on their probabilities.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler

from ..config import load_config, path
from ..features import store
from ..schema import Generation, Label, Question, read_jsonl

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


# ---------- data ----------

def load_labels(cfg: dict, file: str, labels: str) -> dict[str, str]:
    """A label dir from Person 2 (default, default_dense, ...), or a generation run used as an
    ANSWER/NOT_ANSWER proxy."""
    if (p := path(cfg, "labels", labels, f"{file}.jsonl")).exists():
        return {r.qid: r.label for r in read_jsonl(p, Label)}
    gens = read_jsonl(path(cfg, "generations", labels, f"{file}.jsonl"), Generation)
    return {g.qid: "ANSWER" if g.correct else "NOT_ANSWER" for g in gens}


def load_block(cfg: dict, files: list[str], labels: str, layers: list[int], kinds: list[str]):
    """Rows = questions that have both features and a label, in feature order."""
    qids, ys, texts = [], [], []
    feats: dict[str, list] = {f"{k}@L{L}": [] for L in layers for k in kinds}
    for file in files:
        lab = load_labels(cfg, file, labels)
        qtext = {q.qid: q.question for q in read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question)}
        _, fq = store.load(cfg, file, layers[0], kinds[0])
        keep = np.array([q in lab for q in fq])
        qids += [q for q, k in zip(fq, keep) if k]
        ys += [lab[q] for q, k in zip(fq, keep) if k]
        texts += [qtext[q] for q, k in zip(fq, keep) if k]
        for L in layers:
            for k in kinds:
                X, _ = store.load(cfg, file, L, k)
                feats[f"{k}@L{L}"].append(X[keep])
    X = {name: sp.vstack(p).tocsr() if sp.issparse(p[0]) else np.concatenate(p)
         for name, p in feats.items()}
    return qids, np.array(ys), np.array(texts, dtype=object), X


# ---------- models ----------

def make_model(name: str, penalty: str, binarize: bool, seed: int):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    lr = LogisticRegression(
        class_weight="balanced", max_iter=5000, random_state=seed,
        l1_ratio=1.0 if penalty == "l1" else 0.0, solver="saga" if penalty == "l1" else "lbfgs")
    if name == "text-tfidf":
        steps = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr]
    elif name.startswith("sae"):
        steps = [FunctionTransformer(lambda X: (X > 0).astype(np.float32))] if binarize else []
        steps += [MaxAbsScaler(), lr]       # keeps the codes sparse
    else:
        steps = [StandardScaler(), lr]
    pipe = make_pipeline(*steps)
    return GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=3, scoring="f1_macro", n_jobs=-1)


# ---------- metrics ----------

def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def score(p: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = p.argmax(1)
    out = {"macro_f1": f1_score(y, pred, average="macro", labels=range(n_classes)),
           "accuracy": accuracy_score(y, pred), "ece": ece(p, y)}
    try:
        out["auroc"] = (roc_auc_score(y, p[:, 1]) if n_classes == 2
                        else roc_auc_score(y, p, multi_class="ovr", labels=range(n_classes)))
    except ValueError:      # a class missing from this test fold
        out["auroc"] = float("nan")
    return out


def subsample(idx: np.ndarray, y: np.ndarray, budget: int | None, seed: int) -> np.ndarray:
    if budget is None or budget >= len(idx):
        return idx
    sub, _ = train_test_split(idx, train_size=budget, stratify=y[idx], random_state=seed)
    return sub


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="held-out files; default is k-fold CV")
    ap.add_argument("--labels", default="default", help="label variant, or 'closed_hf' for the proxy")
    ap.add_argument("--layers", nargs="+", type=int)
    ap.add_argument("--kinds", nargs="+", help="default: resid_last + sae{w}_last for each width")
    ap.add_argument("--budgets", nargs="+", type=int, default=[], help="label budgets; full is always run")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--penalty", choices=["l1", "l2"], default="l2")
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top-features", type=int, default=10)
    ap.add_argument("--name", default="", help="run-dir suffix, for runs that share train and eval")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = cfg["model"]
    layers = args.layers or m["sae_layers"]
    kinds = args.kinds or ["resid_last"] + [f"sae{w}_last" for w in m["sae_widths"]]

    qids, y_str, texts, feats = load_block(cfg, args.train, args.labels, layers, kinds)
    classes = sorted(set(y_str))
    n_train = len(qids)
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.eval:
        for f in args.eval:
            q2, y2, t2, f2 = load_block(cfg, [f], args.labels, layers, kinds)
            start = len(qids)
            qids += q2
            y_str, texts = np.concatenate([y_str, y2]), np.concatenate([texts, t2])
            for k in feats:
                feats[k] = (sp.vstack([feats[k], f2[k]]).tocsr() if sp.issparse(feats[k])
                            else np.concatenate([feats[k], f2[k]]))
            splits.append((f, np.arange(n_train), np.arange(start, len(qids))))
    y = np.array([classes.index(c) for c in y_str])
    if not args.eval:
        k = min(args.folds, np.bincount(y[:n_train]).min())
        skf = StratifiedKFold(k, shuffle=True, random_state=cfg["seed"])
        splits = [(f"cv{k}", tr, te) for tr, te in skf.split(np.zeros(n_train), y[:n_train])]

    print(f"{n_train} training questions, classes: "
          + ", ".join(f"{c}={np.mean(y_str[:n_train] == c):.0%}" for c in classes))

    models = {"majority": None, "text-tfidf": texts} | feats
    variant = args.penalty + ("_bin" if args.binarize else "")
    run = "+".join(args.train) + ("__" + "+".join(args.eval) if args.eval else "") + f"__{variant}"
    run += f"__{args.name}" if args.name else ""
    out_dir = path(cfg, "results", "probes", m["tag"], args.labels, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, X in models.items():
        X = np.zeros((len(y), 1)) if X is None else X
        for budget in [None, *args.budgets]:
            for seed in args.seeds:
                if budget is None and seed != args.seeds[0]:
                    continue        # the full budget has no sampling, so seeds only repeat it
                probs = np.zeros((len(y), len(classes)))
                per_eval: dict[str, list] = {}
                model = None
                for ev, tr, te in splits:
                    if model is None or not args.eval:      # held-out files share one fit
                        tr = subsample(tr, y, budget, seed)
                        model = make_model(name, args.penalty, args.binarize, seed).fit(X[tr], y[tr])
                    probs[te] = model.predict_proba(X[te])
                    per_eval.setdefault(ev if args.eval else "cv", []).append(te)
                for ev, tes in per_eval.items():
                    te = np.concatenate(tes)
                    rows.append({"model": name, "budget": budget or "full", "seed": seed, "eval": ev,
                                 **score(probs[te], y[te], len(classes))})
                if budget is None and seed == args.seeds[0] and name != "majority":
                    pid = f"{m['tag']}__{args.labels}__{name}__{variant}"
                    rdir = path(cfg, "results", "routers", pid)
                    rdir.mkdir(parents=True, exist_ok=True)
                    for ev, tes in per_eval.items():
                        te = np.concatenate(tes)
                        fname = ev if args.eval else "+".join(args.train)
                        with open(rdir / f"probs_{fname}.jsonl", "w") as f:
                            f.writelines(json.dumps({"qid": qids[i], "p": dict(
                                    zip(classes, map(float, probs[i])))}) + "\n" for i in te)
        print(f"  done {name}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    table = (df.groupby(["eval", "model", "budget"], sort=False)
               [["macro_f1", "auroc", "ece", "accuracy"]].mean().round(3))
    print(table.to_string())

    # Top SAE features per class from one full-data fit (for Neuronpedia lookup).
    for name, X in feats.items():
        if not name.startswith("sae") or args.top_features <= 0:
            continue
        model = make_model(name, args.penalty, args.binarize, args.seeds[0]).fit(X[:n_train], y[:n_train])
        coef = model.best_estimator_[-1].coef_
        coef = np.vstack([-coef[0], coef[0]]) if coef.shape[0] == 1 else coef
        top = {c: [int(i) for i in np.argsort(-coef[j])[: args.top_features]]
               for j, c in enumerate(classes)}
        (out_dir / f"top_features_{name}.json").write_text(json.dumps(top, indent=2))
        print(f"top {name} features:", json.dumps(top))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
