"""BM25 search time per question at batch size 1, single thread (H4, Gate 3).

Uses Pyserini's prebuilt wikipedia-dpr-100w index: the same 21,015,324 DPR passages as
Person 1's ADA index, downloaded instead of built. Times the same questions as
latency_measure.py (the first --n of nq_test and triviaqa_test), after a warm-up, so the
index files are in the page cache as they would be on a running server.

Two timings per question:
    bm25_ms        search top-3 and fetch their text: what a deployed RETRIEVE costs
    bm25_top100_ms search top-100 and fetch them: what the index-hole filter needs (eval only)
Writes runs/latency/bm25.jsonl and runs/latency/bm25_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from saerag.config import load_config, path
from saerag.retrieval.bm25 import BM25Tool
from saerag.schema import Question, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--files", nargs="+", default=["nq_test", "triviaqa_test"])
ap.add_argument("--n", type=int, default=100, help="questions per file (match latency_measure.py)")
ap.add_argument("--index", default="wikipedia-dpr-100w")
args = ap.parse_args()
cfg = load_config(args.config)
k = cfg["retrieval"]["top_k"]

t = time.perf_counter()
tool = BM25Tool(args.index, threads=1)
print(f"index loaded in {time.perf_counter() - t:.1f} s", flush=True)

qs = [q for f in args.files for q in list(read_jsonl(path(cfg, "questions", f"{f}.jsonl"), Question))[: args.n]]
for q in qs:                        # warm-up pass: JIT and page cache
    tool.search(q.question, k)


def timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


rows = [{"qid": q.qid, "bm25_ms": timed(lambda: tool.search(q.question, k)),
         "bm25_top100_ms": timed(lambda: tool.search(q.question, cfg["retrieval"]["hole_fetch_k"]))}
        for q in qs]
out = path(cfg, "results", "latency")
out.mkdir(parents=True, exist_ok=True)
with open(out / "bm25.jsonl", "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
df = pd.DataFrame(rows)
cols = ["bm25_ms", "bm25_top100_ms"]
summary = pd.DataFrame({"p50": df[cols].median(), "p95": df[cols].quantile(0.95), "mean": df[cols].mean()}).round(1)
summary.to_csv(out / "bm25_summary.csv")
print(summary.to_string())
