"""Generation runs for labelling: closed-book, BM25-RAG and gold. Owner: Person 2.

    uv run python -m saerag.generation.run --condition closed --files pilot
    uv run python -m saerag.generation.run --condition bm25   --files pilot

Reads data/questions/{file}.jsonl (+ data/retrieval/{condition}/{file}.jsonl for RAG)
and writes data/generations/{condition}/{file}.jsonl as Generation records, scoring
`correct` with answers.is_correct. Greedy decoding, chat template, vLLM.
"""

from __future__ import annotations

import argparse

from ..config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--condition", choices=["closed", "bm25", "gold", "dense"], required=True)
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # TODO(P2): vllm.LLM(cfg["model"]["generator"]).chat(...) with prompts.closed_book_messages
    # or prompts.rag_messages; SamplingParams(temperature=0, max_tokens=max_new_tokens).
    # Strip the output to the first line before scoring.
    raise NotImplementedError(cfg)


if __name__ == "__main__":
    main()
