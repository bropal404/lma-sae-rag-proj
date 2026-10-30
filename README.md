# SAE-Gated RAG

> **Please go through [TODO.md](TODO.md)** for who is doing what.

A frozen LLM reads only the question. A probe on its SAE features (or on the dense activation) picks **ANSWER**, **RETRIEVE** or **ABSTAIN** before any retrieval or generation happens.
The proposal is the PDF in this directory. The work plan is in [docs/PLAN.md](docs/PLAN.md), and the choices the code assumes are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

```bash
uv sync                      # core: data, labels, probes, metrics (CPU)
uv sync --extra retrieval    # + Pyserini BM25 (needs Java 21+), FAISS
uv sync --extra gpu          # + torch, vLLM, transformers, sae-lens (GPU node)
uv run pytest
```

Point data at a shared directory with `export SAERAG_DATA=/path/to/shared/data` (default `./data`, gitignored).

## Pipeline

| Step | Command | Output | Owner |
|------|---------|--------|-------|
| Splits | `uv run python -m saerag.data.build` | `data/questions/{dataset}_{split}.jsonl`, `pilot.jsonl` | done |
| Retrieval | `python -m saerag.retrieval.run --files pilot` | `data/retrieval/{bm25,gold}/` | P1 |
| Generation | `python -m saerag.generation.run --condition closed --files pilot` | `data/generations/{condition}/` | P2 |
| Labels | `python -m saerag.generation.make_labels --files pilot` | `data/labels/{variant}/` | P2 |
| Features | `python -m saerag.features.extract --files pilot` | `data/features/{tag}/L{layer}/` | P3 |
| Probes | `python -m saerag.probes.train` | `runs/routers/<id>/probs_*.jsonl` | P3 |
| Baselines | `python -m saerag.baselines.run --routers b1 b2 b3 b8 --files pilot` | `runs/routers/<id>/` | P4 |
| Scores | `python -m saerag.eval.score --routers <ids> --files <files> --val-file nq_val` | `runs/scores/` | P4 |
| Hand check | `python -m saerag.eval.handcheck sample --files ...` then `score` ([protocol](docs/HANDCHECK.md)) | `runs/handcheck/` | P4 |

Shared, tested contract (change only after telling the team): `schema.py` (record formats), `answers.py` (correctness), `labels.py` (label rule), `prompts.py` (frozen at pilot).

## Git
Work on a branch per stream (`p1-retrieval`, `p2-labels`, `p3-probes`, `p4-eval`) and merge into `main` through small PRs. Never commit data, features or model files.
