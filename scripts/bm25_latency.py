"""BM25 search latency per question at batch size 1, for Gate 3 / H4. Owner: Person 2. CPU only.

Each question is searched alone with one thread, passage fetching included, at k = top_k (what a
deployed router fetches) and k = hole_fetch_k (what the hole filter fetched). The index is
Pyserini's prebuilt wikipedia-dpr-100w: the same 21,015,324-passage corpus as the team's index.
The first run downloads it (~10 GB) to ~/.cache/pyserini. Needs Java 21+ and pyserini.

    PYTHONPATH=src python scripts/bm25_latency.py [--index DIR] [--files nq_test triviaqa_test] [--n 200]

Writes runs/latency/bm25_search.jsonl, bm25_summary.csv and bm25_latency.md.
"""

import argparse
import json
import os
import platform
import subprocess
import time

import pandas as pd

from saerag.config import load_config, path
from saerag.retrieval.bm25 import BM25Tool
from saerag.schema import Question, read_jsonl

N_DOCS = 21_015_324


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        return next(line.split(":", 1)[1].strip() for line in out.splitlines() if "Model name" in line)
    except Exception:
        return platform.processor() or "unknown CPU"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--index", help="local Lucene index directory (default: the prebuilt index)")
    ap.add_argument("--files", nargs="+", default=["nq_test", "triviaqa_test"])
    ap.add_argument("--n", type=int, default=200, help="questions per file")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    k_deploy, k_fetch = cfg["retrieval"]["top_k"], cfg["retrieval"]["hole_fetch_k"]

    t0 = time.time()
    tool = BM25Tool(args.index or cfg["retrieval"]["bm25_index"], threads=1)
    n_docs = tool.searcher.num_docs
    print(f"index ready in {(time.time() - t0) / 60:.1f} min, {n_docs:,} passages", flush=True)
    assert n_docs == N_DOCS, f"expected the {N_DOCS:,}-passage DPR Wikipedia corpus"

    def timed(q: str, k: int) -> tuple[float, int]:
        t = time.perf_counter()
        hits = tool.search(q, k)
        return (time.perf_counter() - t) * 1000, len(hits)

    rows = []
    for f in args.files:
        qs = list(read_jsonl(path(cfg, "questions", f"{f}.jsonl"), Question))[: args.n]
        for q in qs[: args.warmup]:          # JVM warm-up and page cache
            tool.search(q.question, k_fetch)
        for q in qs:
            ms_d, hits = timed(q.question, k_deploy)
            ms_f, _ = timed(q.question, k_fetch)
            rows.append({"qid": q.qid, "file": f, f"search_k{k_deploy}_ms": ms_d,
                         f"search_k{k_fetch}_ms": ms_f, "hits": hits})
        print(f"{f}: searched {len(qs)} questions", flush=True)

    out = path(cfg, "results", "latency")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bm25_search.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    df = pd.DataFrame(rows)
    cols = [f"search_k{k_deploy}_ms", f"search_k{k_fetch}_ms"]
    summary = pd.DataFrame({"p50": df[cols].median(), "p95": df[cols].quantile(0.95),
                            "mean": df[cols].mean(), "n": df[cols].count()}).round(1)
    summary.to_csv(out / "bm25_summary.csv")
    text = "\n".join([
        "# BM25 search latency at batch size 1", "",
        f"{len(rows)} questions ({', '.join(args.files)}), one query at a time, one search thread, "
        f"passage fetching included. {cpu_name()}, {os.cpu_count()} logical CPUs. "
        f"Index: {n_docs:,} passages.", "",
        "```", summary.to_string(), "```", "",
        f"A routed RETRIEVE costs BM25 search (k = {k_deploy}) + RAG generation.", ""])
    (out / "bm25_latency.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
