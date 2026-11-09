"""Analyses that need the per-question files from the SageMaker bundle (runs/, data/).

    uv run python scripts/analysis/bundle_analysis.py

Writes results/analysis/:
    sae_vs_dense_bootstrap.csv  paired bootstrap of SAE minus dense, same layer, same questions
    confusion.csv               confusion matrices (rows = label, columns = action)
    popqa_popularity.csv        PopQA by subject-popularity quartile
    routing_errors.csv          routing mistakes on NQ / TriviaQA test, for hand annotation
Routing uses the ABSTAIN threshold tuned on val, as in eval.score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from saerag.config import REPO_ROOT, load_config, path
from saerag.eval import metrics
from saerag.eval.score import load_labels, load_probs, threshold_from_val
from saerag.schema import Generation, Question, read_jsonl

VARIANT = "miss_as_retrieve"
TAG = "gemma3-4b-it"
TESTS = ["nq_test", "triviaqa_test", "popqa_test", "hotpotqa_test", "selfaware_test"]
LAYERS = [17, 22, 29]
N_BOOT = 2000


def pid(kind: str, layer: int) -> str:
    return f"{TAG}__{VARIANT}__{kind}@L{layer}__l2"


def val_for(file: str) -> str:
    return "triviaqa_val" if file.startswith("triviaqa") else "nq_val"


def routed(cfg, rid: str, file: str):
    probs, _ = load_probs(cfg, rid, file)
    t = threshold_from_val(cfg, rid, val_for(file), VARIANT)
    return probs, metrics.route(probs, t)


def stats(acts, labels, closed, rag, qids) -> dict[str, float]:
    a = {q: acts[q] for q in qids}
    out = metrics.outcomes(a, {q: closed[q] for q in qids}, {q: rag[q] for q in qids})
    return {"macro_f1": metrics.macro_f1(a, {q: labels[q] for q in qids}),
            "accuracy": metrics.accuracy(out),
            "hallucination": metrics.hallucination_rate(out),
            "retrieval_calls": metrics.retrieval_call_rate(a)}


def main() -> None:
    cfg = load_config()
    out_dir = REPO_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    boot, conf, pop_rows, errors = [], [], [], []

    for file in TESTS:
        qs = {q.qid: q for q in read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question)}
        lab = load_labels(cfg, file, VARIANT)
        labels = {q: r.label for q, r in lab.items()}
        closed = {q: r.closed_correct for q, r in lab.items()}
        rag = {q: r.rag_correct for q, r in lab.items()}
        qids = np.array(sorted(labels))

        acts = {}
        for L in LAYERS:
            for kind in ("sae16k_last", "resid_last"):
                _, acts[kind, L] = routed(cfg, pid(kind, L), file)

        # Paired bootstrap: resample questions once, score both probes on the same draw.
        for L in LAYERS:
            sae, dense = acts["sae16k_last", L], acts["resid_last", L]
            point = {k: stats(sae, labels, closed, rag, qids)[k] - stats(dense, labels, closed, rag, qids)[k]
                     for k in ("macro_f1", "accuracy", "hallucination", "retrieval_calls")}
            draws = {k: [] for k in point}
            for _ in range(N_BOOT):
                s = rng.choice(qids, len(qids))
                a, b = stats(sae, labels, closed, rag, s), stats(dense, labels, closed, rag, s)
                for k in draws:
                    if not (np.isnan(a[k]) or np.isnan(b[k])):
                        draws[k].append(a[k] - b[k])
            for k, d in draws.items():
                lo, hi = np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)
                boot.append({"file": file, "layer": L, "metric": k, "sae_minus_dense": point[k],
                             "ci_low": lo, "ci_high": hi, "significant": not (lo <= 0 <= hi)})

        for kind in ("sae16k_last", "resid_last"):
            cm = metrics.confusion(acts[kind, 22], labels)
            for true, row in cm.items():
                conf.append({"file": file, "router": f"{kind}@L22", "label": true, **row})

        if file == "popqa_test":
            spop = pd.Series({q: qs[q].meta.get("s_pop") for q in qids}, dtype=float)
            bins = pd.qcut(spop, 4, labels=["Q1 (rarest)", "Q2", "Q3", "Q4 (most popular)"])
            for b in bins.cat.categories:
                sub = list(bins.index[bins == b])
                row = {"bin": b, "n": len(sub), "s_pop_min": spop[sub].min(), "s_pop_max": spop[sub].max(),
                       "closed_acc": np.mean([closed[q] for q in sub]),
                       "bm25_acc": np.mean([rag[q] for q in sub])}
                for a in ("ANSWER", "RETRIEVE", "ABSTAIN"):
                    row[f"label_{a}"] = np.mean([labels[q] == a for q in sub])
                for kind in ("sae16k_last", "resid_last"):
                    s = stats(acts[kind, 22], labels, closed, rag, sub)
                    row |= {f"{kind}@L22_{k}": v for k, v in s.items()}
                pop_rows.append(row)

        if file in ("nq_test", "triviaqa_test"):
            gens = {c: {g.qid: g.output for g in read_jsonl(path(cfg, "generations", c, f"{file}.jsonl"), Generation)}
                    for c in ("closed", "bm25")}
            a = acts["sae16k_last", 22]
            wrong = [q for q in qids if a[q] != labels[q]]
            pick = rng.choice(wrong, min(15, len(wrong)), replace=False)
            for q in pick:
                errors.append({"file": file, "qid": q, "hole": qs[q].hole, "question": qs[q].question,
                               "answers": " | ".join(qs[q].answers[:5]), "label": labels[q],
                               "sae@L22_action": a[q], "dense@L22_action": acts["resid_last", 22][q],
                               "closed_output": gens["closed"].get(q), "bm25_output": gens["bm25"].get(q),
                               "note": ""})

    pd.DataFrame(boot).round(4).to_csv(out_dir / "sae_vs_dense_bootstrap.csv", index=False)
    pd.DataFrame(conf).to_csv(out_dir / "confusion.csv", index=False)
    pd.DataFrame(pop_rows).round(3).to_csv(out_dir / "popqa_popularity.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "routing_errors.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
