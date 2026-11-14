"""Compose final H4 latency using measured SAE L29 and dense L22 costs.

- Uses the *validation-selected* SAE L29 and dense L22 3-way controllers.
- Uses validation-tuned thresholds, not plain argmax.
- Uses original per-question BM25 timings, not an assumed zero.
- Requires an action for every timed question; never silently drops rows.

This is a composed measured-stage estimate, not a directly timed production server.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from saerag.config import load_config, path
from saerag.eval.metrics import route
from saerag.eval.score import threshold_from_val
from saerag.schema import read_jsonl


def cost_ms(action: str, stage: dict, controller_col: str | None) -> float:
    """Single question, with controller cost even on ABSTAIN."""
    controller = float(stage[controller_col]) if controller_col else 0.0
    if action == "ANSWER":
        return controller + float(stage["closed_ms"])
    if action == "RETRIEVE":
        return controller + float(stage["bm25_ms"]) + float(stage["rag_ms"])
    if action == "ABSTAIN":
        return controller
    raise ValueError(f"Unrecognized routing action: {action}")


def selected_routers(tag: str, variant: str, sae_layer: int, dense_layer: int):
    sae = f"{tag}__{variant}__sae16k_last@L{sae_layer}__l2"
    dense = f"{tag}__{variant}__resid_last@L{dense_layer}__l2"
    baseline = ["b1_always_retrieve", "b2_never_retrieve", "b3_oracle",
                "b4_verbalised", "b8_tfidf_lr"]
    return sae, dense, [*baseline, sae, dense]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="miss_as_retrieve")
    ap.add_argument("--sae-layer", type=int, default=29)
    ap.add_argument("--dense-layer", type=int, default=22)
    ap.add_argument("--stage-file", default="stages_l29.jsonl")
    ap.add_argument("--out", default="routed_l29.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tag = cfg["model"]["tag"]
    sae, dense, routers = selected_routers(tag, a.variant,
                                           a.sae_layer, a.dense_layer)
    stagefile = path(cfg, "results", "latency", a.stage_file)
    stages = pd.read_json(stagefile, lines=True)
    needed = {"qid", "file", "sae_controller_ms", "dense_controller_ms",
              "closed_ms", "rag_ms", "b4_ms"}
    if not needed.issubset(stages.columns):
        raise ValueError(f"Stage file is missing columns: {needed - set(stages.columns)}")
    if stages.empty or stages.qid.duplicated().any():
        raise ValueError("Stage file is empty or contains duplicate question IDs")
    if not stages.file.isin(["nq_test", "triviaqa_test"]).all():
        raise ValueError("Expected NQ and TriviaQA test questions only")
    stages = stages.set_index("qid")
    bm25file = path(cfg, "results", "latency", "bm25.jsonl")
    if not bm25file.exists():
        alt = path(cfg, "results").parent / "results/sagemaker/latency/bm25.jsonl"
        if not alt.exists():
            raise FileNotFoundError(f"BM25 timings not found at {bm25file}")
        bm25file = alt
    bm25 = pd.read_json(bm25file, lines=True)
    if bm25.qid.duplicated().any():
        raise ValueError("Duplicate QIDs in BM25 timings")
    bm25 = bm25.set_index("qid")
    absent = stages.index.difference(bm25.index)
    if len(absent):
        raise ValueError(f"Missing BM25 timings for {len(absent)} measured questions")
    stages["bm25_ms"] = bm25.loc[stages.index, "bm25_ms"]
    time_cols = ["sae_controller_ms", "dense_controller_ms", "closed_ms",
                 "rag_ms", "b4_ms", "bm25_ms"]
    if not np.isfinite(stages[time_cols].to_numpy(dtype=float)).all() or \
       (stages[time_cols].to_numpy(dtype=float) < 0).any():
        raise ValueError("Timing columns must be finite nonnegative milliseconds")

    out = []
    for router_id in routers:
        actions = {}
        for file, block in stages.groupby("file", sort=False):
            p = path(cfg, "results", "routers", router_id,
                     f"probs_{file}.jsonl")
            if not p.is_file():
                raise FileNotFoundError(f"Missing router probabilities: {p}")
            timed_ids = set(block.index)
            probs = {r["qid"]: r["p"] for r in read_jsonl(p)
                     if r["qid"] in timed_ids}
            if set(probs) != timed_ids:
                raise ValueError(f"{router_id}/{file}: have {len(probs)} of "
                                 f"{len(timed_ids)} measured QIDs")
            val = "triviaqa_val" if file == "triviaqa_test" else "nq_val"
            threshold = threshold_from_val(cfg, router_id, val, a.variant)
            if threshold is None:
                raise ValueError(f"No validation threshold for {router_id}/{val}")
            actions.update(route(probs, threshold))
        if set(actions) != set(stages.index):
            raise ValueError(f"{router_id}: missing actions for timed question IDs")
        column = ("sae_controller_ms" if router_id == sae else
                  "dense_controller_ms" if router_id == dense else
                  "b4_ms" if router_id == "b4_verbalised" else None)
        costs = [cost_ms(actions[qid], stages.loc[qid], column)
                 for qid in stages.index]
        out.append({"router": router_id, "n": len(costs),
                    "retrieval_calls": float(np.mean([a == "RETRIEVE" for a in actions.values()])),
                    "p50_ms": float(np.median(costs)),
                    "p95_ms": float(np.percentile(costs, 95)),
                    "mean_ms": float(np.mean(costs)),
                    "routing_policy": "validation-threshold",
                    "sae_layer": a.sae_layer, "dense_layer": a.dense_layer,
                    "method": "composed_measured_stages"})
    frame = pd.DataFrame(out).round(3)
    output = path(cfg, "results", "latency", a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print("WROTE", output)


if __name__ == "__main__":
    main()
