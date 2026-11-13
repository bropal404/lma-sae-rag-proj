"""Per-router timing at batch size 1 on identical questions.

The --match-scoring flag applies the *same validation-tuned abstain threshold* as
saerag.eval.score; without it the original script used argmax, making the
latency and reported-accuracy operating points incomparable. It is recommended
for final H4 analysis. Specify --stage-file and --out to compare probe layers
without overwriting the existing stage measurements.

    python scripts/sagemaker/latency_route.py --config configs/sagemaker.yaml \
      --variant miss_as_retrieve --layer 29 --stage-file stages_l29.jsonl \
      --out routed_l29.csv --match-scoring
"""
import argparse

import numpy as np
import pandas as pd

from saerag.config import load_config, path
from saerag.eval.metrics import argmax_actions, route
from saerag.eval.score import threshold_from_val
from saerag.schema import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--variant", required=True)
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--stage-file", default="stages.jsonl")
ap.add_argument("--out", default="routed.csv", help="filename under runs/latency/")
ap.add_argument("--match-scoring", action="store_true", help="apply validation-tuned thresholds")
args = ap.parse_args()
cfg = load_config(args.config)
tag = cfg["model"]["tag"]

stages = pd.read_json(path(cfg, "results", "latency", args.stage_file), lines=True).set_index("qid")
if stages.index.has_duplicates:
    raise ValueError("duplicate qids in stage measurements")
bm25 = path(cfg, "results", "latency", "bm25.jsonl")
if bm25.exists():
    times = pd.read_json(bm25, lines=True).set_index("qid").bm25_ms
    if not set(stages.index).issubset(set(times.index)):
        raise ValueError("missing BM25 per-question timings; rerun bm25_latency.py on matching questions")
    stages["bm25_ms"] = times
else:
    raise FileNotFoundError("BM25 latency missing. H4 must include retrieval time, not assume zero.")

probes = [f"{tag}__{args.variant}__{k}@L{args.layer}__l2{s}"
          for k in ("resid_last", "sae16k_last") for s in ("", "__2way")]
routers = ["b1_always_retrieve", "b2_never_retrieve", "b3_oracle",
           "b4_verbalised", "b8_tfidf_lr", *probes]

rows = []
for rid in routers:
    all_actions = {}
    for f in stages.file.unique():
        p = path(cfg, "results", "routers", rid, f"probs_{f}.jsonl")
        if not p.exists():
            print("no probabilities for", rid, f)
            continue
        within_file = set(stages.index[stages.file == f])
        probs = {r["qid"]: r["p"] for r in read_jsonl(p) if r["qid"] in within_file}
        if not probs:
            continue
        if args.match_scoring:
            val = "triviaqa_val" if f.startswith("triviaqa") else "nq_val"
            t = threshold_from_val(cfg, rid, val, args.variant)
            if t is None:
                raise ValueError(f"Missing validation probabilities for {rid}/{val}")
            acts = route(probs, t)
        else:
            acts = argmax_actions(probs)
        all_actions.update(acts)
    if not all_actions:
        continue
    if len(all_actions) != len(stages):
        raise ValueError(f"{rid}: have {len(all_actions)} actions for {len(stages)} timed questions")
    ctrl = "controller_ms" if rid in probes else ("b4_ms" if rid == "b4_verbalised" else None)
    lat = []
    for q, a in all_actions.items():
        s = stages.loc[q]
        lat.append((s[ctrl] if ctrl else 0.0) +
                   {"ANSWER": s.closed_ms, "RETRIEVE": s.bm25_ms + s.rag_ms, "ABSTAIN": 0.0}[a])
    rows.append({"router": rid, "n": len(lat),
                 "retrieval_calls": np.mean([a == "RETRIEVE" for a in all_actions.values()]),
                 "p50_ms": np.median(lat), "p95_ms": np.percentile(lat, 95),
                 "mean_ms": np.mean(lat),
                 "routing_policy": "validation-threshold" if args.match_scoring else "argmax",
                 "layer": args.layer})
df = pd.DataFrame(rows).round(3)
output = path(cfg, "results", "latency", args.out)
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)
print(df.to_string(index=False))
print("WROTE", output)
