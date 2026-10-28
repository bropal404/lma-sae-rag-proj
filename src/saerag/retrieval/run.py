"""Offline retrieval for the labelling runs. Owner: Person 1.

    uv run python -m saerag.retrieval.run --files pilot nq_train ...

For each question file writes, keyed by qid:
    data/retrieval/bm25/{file}.jsonl   top_k passages (hole-filtered when q.hole)
    data/retrieval/gold/{file}.jsonl   gold condition passages (missing qid = no gold)
"""

from __future__ import annotations

import argparse

from ..config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # TODO(P1): BM25Tool.batch_search with k = retrieval.hole_fetch_k for every question,
    # then apply_hole for hole questions / plain top_k otherwise, plus gold_passages.
    # Record per-query latency_ms for the latency benchmark.
    raise NotImplementedError(cfg)


if __name__ == "__main__":
    main()
