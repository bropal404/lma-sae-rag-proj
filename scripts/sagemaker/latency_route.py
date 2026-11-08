"""Per-router latency on the timed questions (H4), with argmax routing.

ANSWER costs a closed-book generation, RETRIEVE a BM25-RAG generation (plus BM25 search,
measured separately on ADA), ABSTAIN nothing. Probes add the controller time, B4 its own
generation, and B8 nothing measurable. Writes runs/latency/routed.csv.
"""
import argparse, json

import numpy as np, pandas as pd

from saerag.config import load_config, path
from saerag.eval.metrics import argmax_actions
from saerag.schema import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--variant", required=True)
ap.add_argument("--layer", type=int, default=17)
args = ap.parse_args()
cfg = load_config(args.config)
tag = cfg["model"]["tag"]

stages = pd.read_json(path(cfg, "results", "latency", "stages.jsonl"), lines=True).set_index("qid")
probes = [f"{tag}__{args.variant}__{k}@L{args.layer}__l2{s}" for k in ("resid_last", "sae16k_last") for s in ("", "__2way")]
routers = ["b1_always_retrieve", "b2_never_retrieve", "b3_oracle", "b4_verbalised", "b8_tfidf_lr", *probes]

rows = []
for rid in routers:
    probs = {}
    for f in stages.file.unique():
        p = path(cfg, "results", "routers", rid, f"probs_{f}.jsonl")
        if p.exists():
            probs |= {r["qid"]: r["p"] for r in read_jsonl(p) if r["qid"] in stages.index}
    if not probs:
        print("no probabilities for", rid); continue
    acts = argmax_actions(probs)
    ctrl = "controller_ms" if rid in probes else ("b4_ms" if rid == "b4_verbalised" else None)
    lat = []
    for q, a in acts.items():
        s = stages.loc[q]
        lat.append((s[ctrl] if ctrl else 0.0) + {"ANSWER": s.closed_ms, "RETRIEVE": s.rag_ms, "ABSTAIN": 0.0}[a])
    rows.append({"router": rid, "n": len(lat), "retrieval_calls": np.mean([a == "RETRIEVE" for a in acts.values()]),
                 "p50_ms": np.median(lat), "p95_ms": np.percentile(lat, 95), "mean_ms": np.mean(lat)})
df = pd.DataFrame(rows).round(3)
df.to_csv(path(cfg, "results", "latency", "routed.csv"), index=False)
print(df.to_string(index=False))
