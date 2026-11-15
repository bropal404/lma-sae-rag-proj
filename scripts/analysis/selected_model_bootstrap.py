#!/usr/bin/env python3
"""Paired, dataset-stratified bootstrap for validation-selected SAE L29 vs dense L22.

Uses canonical labels and per-question probability JSONL files. The router operating
point matches saerag.eval.score: abstention threshold tuned on validation only.
This is a *routed-action* comparison, not the probe grid's argmax-only F1.
No model execution/GPU is required. Never changes labels, features, or probes.

From repo root:
  PYTHONPATH=src python scripts/analysis/selected_model_bootstrap.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from saerag.config import REPO_ROOT, load_config
from saerag.eval import metrics
from saerag.eval.score import load_labels, load_probs, threshold_from_val
from saerag.schema import ACTIONS

VARIANT = "miss_as_retrieve"
SAE = "gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2"
DENSE = "gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2"
FILES = ("nq_test", "triviaqa_test", "popqa_test", "hotpotqa_test")
INDEX = {name: ix for ix, name in enumerate(ACTIONS)}


def prepare(cfg, file):
    labels = load_labels(cfg, file, VARIANT)
    if len(labels) != 1000:
        raise ValueError(f"{file}: expected 1000 labels, found {len(labels)}")
    qids = sorted(labels)
    val = "triviaqa_val" if file.startswith("triviaqa") else "nq_val"
    actions = []
    for rid in (SAE, DENSE):
        probs, _ = load_probs(cfg, rid, file)
        if probs is None or set(probs) != set(labels):
            raise ValueError(f"{file}: incomplete probabilities for {rid}")
        t = threshold_from_val(cfg, rid, val, VARIANT)
        if t is None:
            raise ValueError(f"{file}: missing validation probabilities for {rid}/{val}")
        acts = metrics.route(probs, t)
        actions.append(np.asarray([INDEX[acts[q]] for q in qids], dtype=np.int8))
    truth = np.asarray([INDEX[labels[q].label] for q in qids], dtype=np.int8)
    closed = np.asarray([labels[q].closed_correct for q in qids], dtype=bool)
    rag = np.asarray([labels[q].rag_correct for q in qids], dtype=bool)
    return truth, actions[0], actions[1], closed, rag


def evaluate_arrays(y, a, closed, rag):
    """Same class-presence convention as saerag.eval.metrics.macro_f1."""
    conf = np.bincount(3 * y.astype(int) + a.astype(int), minlength=9).reshape(3, 3)
    support = conf.sum(1)
    predicted = conf.sum(0)
    relevant = (support + predicted) > 0
    denom = support + predicted
    scores = np.divide(2 * np.diag(conf), denom, out=np.zeros(3, dtype=float), where=denom > 0)
    correct = np.where(a == INDEX["ANSWER"], closed,
                       np.where(a == INDEX["RETRIEVE"], rag, False))
    return {"macro_f1": float(scores[relevant].mean()),
            "accuracy": float(correct.mean()),
            "retrieval_calls": float((a == INDEX["RETRIEVE"]).mean())}


def compare(x, indices):
    y, sae, dense, closed, rag = x
    return {m: evaluate_arrays(y[indices], sae[indices], closed[indices], rag[indices])[m]
            - evaluate_arrays(y[indices], dense[indices], closed[indices], rag[indices])[m]
            for m in ("macro_f1", "accuracy", "retrieval_calls")}


def result(file, metric, point, vals, n, draws):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"file": file, "n": n, "metric": metric, "sae_minus_dense": point,
            "ci_low": float(lo), "ci_high": float(hi),
            "significant_vs_zero": bool(lo > 0 or hi < 0),
            "draws": draws, "sae": "sae16k_last@L29", "dense": "resid_last@L22"}


def run(cfg, draws, seed):
    data = {file: prepare(cfg, file) for file in FILES}
    rng = np.random.default_rng(seed)
    metrics_list = ("macro_f1", "accuracy", "retrieval_calls")
    per_file = {file: {m: np.empty(draws, dtype=float) for m in metrics_list} for file in FILES}
    aggregate = {m: np.empty(draws, dtype=float) for m in metrics_list}
    rows = []
    points = {file: compare(d, np.arange(len(d[0]))) for file, d in data.items()}
    for iteration in range(draws):
        per_iter = []
        for file, d in data.items():
            ids = rng.integers(0, len(d[0]), size=len(d[0]))
            diff = compare(d, ids)
            for m in metrics_list:
                per_file[file][m][iteration] = diff[m]
            per_iter.append(diff)
        for m in metrics_list:
            # Dataset-stratified bootstrap. All 4 datasets have n=1000, so mean
            # per-dataset accuracy also equals micro-weighted accuracy here.
            aggregate[m][iteration] = float(np.mean([d[m] for d in per_iter]))
    for file, d in data.items():
        for m in metrics_list:
            rows.append(result(file, m, points[file][m], per_file[file][m], len(d[0]), draws))
    for m in metrics_list:
        rows.append(result("answerable_test_mean_4", m,
                           float(np.mean([p[m] for p in points.values()])),
                           aggregate[m], sum(len(d[0]) for d in data.values()), draws))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/analysis/selected_model_bootstrap.csv")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results/analysis/selected_model_bootstrap.md")
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("Use >=100 draws; default is 2000")
    df = run(load_config(args.config), args.draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(df[["file", "metric", "sae_minus_dense", "ci_low", "ci_high"]].round(4).to_string(index=False))
    p = df[(df.file == "answerable_test_mean_4") & (df.metric == "macro_f1")].iloc[0]
    print(f"\nPrimary routed macro-F1 difference = {p.sae_minus_dense:+.4f}, "
          f"paired stratified 95% percentile CI [{p.ci_low:+.4f}, {p.ci_high:+.4f}]")
    print("At-least-minus-two-points point criterion:", bool(p.sae_minus_dense >= -0.02))
    print("95% CI entirely above -0.02:", bool(p.ci_low > -0.02))
    print("Note: This CI uses validation-tuned routing decisions and is NOT "
          "the raw argmax-probe-grid macro-F1 reported in H2.")
    parts = ["# Selected-model paired bootstrap", "", "Validation-selected SAE `sae16k_last@L29` versus dense `resid_last@L22`.", "Decisions use the same validation-only abstention threshold as `saerag.eval.score`.", "", "| Dataset | SAE − dense macro-F1 | 95% percentile CI |", "|---|---:|---:|"]
    for _, row in df[df.metric == "macro_f1"].iterrows():
        parts.append(f"| {row.file} | {row.sae_minus_dense*100:+.2f} pp | [{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}] pp |")
    parts += ["", "The answerable-test aggregate is the unweighted mean of four 1,000-question dataset macro-F1 scores; each bootstrap draw resamples questions **within each dataset** and preserves the SAE/dense pairing.", "",
              "The routed point estimate is inside the -2 pp tolerance only if it is >= -2 pp. A 95% CI that extends below -2 pp **does not establish** the non-inferiority margin.",
              "This routed-action metric is distinct from the raw probe-grid H2 macro-F1, which may use different decision rules; report both with explicit labels.", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(parts))
    print("WROTE", args.output, "and", args.report)


if __name__ == "__main__":
    main()
