"""Score every router on the same questions with the same metrics. Owner: Person 4.

    uv run python -m saerag.eval.score --routers b1_always_retrieve b3_oracle my_probe \
        --files nq_test triviaqa_test popqa_test --val-file nq_val [--latency]

A router id is looked up in runs/routers/{id}/ and then runs/probes/{id}/. With
--val-file, each router's ABSTAIN threshold is tuned on that file for at most
operating_points.max_false_refusal of answerable questions refused; without it,
routing is the plain argmax.
"Should abstain" means label == ABSTAIN, so abstention recall covers Self-Aware and
the index-hole questions the model cannot answer, as the proposal defines it.
Every metric is reported on all questions, hole and non-hole questions (D3) and per
dataset. Writes runs/scores/{variant}/{file}.csv.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..config import load_config, path
from ..schema import Generation, Label, Question, Retrieval, read_jsonl
from . import metrics


def find_probs(cfg, router_id: str, file: str):
    for kind in ("routers", "probes"):
        p = path(cfg, "results", kind, router_id, f"probs_{file}.jsonl")
        if p.exists():
            return p
    return None


def load_probs(cfg, router_id: str, file: str):
    p = find_probs(cfg, router_id, file)
    if p is None:
        return None, None
    rows = list(read_jsonl(p))
    lat = {r["qid"]: r["latency_ms"] for r in rows if r.get("latency_ms") is not None}
    return {r["qid"]: r["p"] for r in rows}, lat


def load_labels(cfg, file: str, variant: str) -> dict[str, Label]:
    return {r.qid: r for r in read_jsonl(path(cfg, "labels", variant, f"{file}.jsonl"), Label)}


def subsets(questions: dict[str, Question], qids) -> dict[str, list[str]]:
    qids = [q for q in qids if q in questions]
    out = {"all": qids,
           "hole": [q for q in qids if questions[q].hole],
           "no_hole": [q for q in qids if not questions[q].hole]}
    for ds in sorted({questions[q].dataset for q in qids}):
        out[ds] = [q for q in qids if questions[q].dataset == ds]
    return {k: v for k, v in out.items() if v}


def threshold_from_val(cfg, router_id: str, val_file: str, variant: str) -> float | None:
    probs, _ = load_probs(cfg, router_id, val_file)
    if probs is None:
        return None
    labels = load_labels(cfg, val_file, variant)
    p_abs = {q: probs[q]["ABSTAIN"] for q in labels if q in probs}
    should = {q: labels[q].label == "ABSTAIN" for q in p_abs}
    return metrics.tune_abstain_threshold(p_abs, should,
                                          cfg["operating_points"]["max_false_refusal"])


def stage_latencies(cfg, file: str):
    """qid -> ms for retrieval and both generation runs, or None if any is missing."""
    def by_qid(kind, sub, cls):
        p = path(cfg, kind, sub, f"{file}.jsonl")
        if not p.exists():
            return None
        d = {r.qid: r.latency_ms for r in read_jsonl(p, cls)}
        return None if any(v is None for v in d.values()) else d

    ret = by_qid("retrieval", "bm25", Retrieval)
    closed = by_qid("generations", "closed", Generation)
    rag = by_qid("generations", "bm25", Generation)
    return None if None in (ret, closed, rag) else (ret, closed, rag)


def score_file(cfg, routers: list[str], file: str, variant: str, val_file: str | None,
               latency: bool) -> pd.DataFrame:
    questions = {q.qid: q for q in read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question)}
    labels = load_labels(cfg, file, variant)
    lat_inputs = stage_latencies(cfg, file) if latency else None
    if latency and lat_inputs is None:
        print(f"{file}: stage latencies missing or incomplete, skipping latency")
    rows = []
    for rid in routers:
        probs, ctrl_ms = load_probs(cfg, rid, file)
        if probs is None:
            print(f"{file}: no probs for {rid}, skipped")
            continue
        t = threshold_from_val(cfg, rid, val_file, variant) if val_file else None
        for name, qids in subsets(questions, labels).items():
            qids = [q for q in qids if q in probs]
            if not qids:
                continue
            row = {"router": rid, "subset": name, "abstain_threshold": t, **metrics.score(
                {q: probs[q] for q in qids},
                {q: labels[q].label for q in qids},
                {q: labels[q].closed_correct for q in qids},
                {q: labels[q].rag_correct for q in qids},
                {q: labels[q].label == "ABSTAIN" for q in qids},
                abstain_threshold=t)}
            if lat_inputs:
                acts = metrics.route({q: probs[q] for q in qids}, t)
                lat = metrics.route_latency(acts, ctrl_ms, *lat_inputs)
                for stage, pct in metrics.latency_percentiles(lat).items():
                    row[f"{stage}_p50_ms"], row[f"{stage}_p95_ms"] = pct["p50"], pct["p95"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--routers", nargs="+", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--val-file", help="tune ABSTAIN thresholds here; omit for argmax")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for f in args.files:
        df = score_file(cfg, args.routers, f, args.variant, args.val_file, args.latency)
        out = path(cfg, "results", "scores", args.variant, f"{f}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        cols = ["router", "subset", "n", "accuracy", "hallucination_rate", "retrieval_call_rate",
                "abstention_recall", "false_refusal_rate", "aurc", "macro_f1"]
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(f"\n== {f} ({args.variant}) -> {out}")
            print(df[[c for c in cols if c in df]].to_string(index=False) if len(df) else "empty")


if __name__ == "__main__":
    main()
