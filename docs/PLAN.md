# Work plan

Owners follow Manya's split. Fill in names. Weeks follow the proposal's timeline; the deadline is still open.

| Person | Owner | Work stream | Code |
|--------|-------|-------------|------|
| 1 | _ | BM25 index, search tool, index holes, gold condition, later the dense retriever | `retrieval/` |
| 2 | _ | Generation (closed / BM25 / gold), answer matching, labels | `generation/`, `answers.py`, `labels.py`, `prompts.py` |
| 3 | _ | SAE coverage check, activations + SAE codes, SAE and dense probes | `features/`, `probes/` |
| 4 | _ | B1–B4, B8, metrics, 200-label hand check | `baselines/`, `eval/` |

## Critical path

```
D1 model check (P3) ──┐
BM25 up (P1) ─────────┼─> pilot retrieval (P1) ─> pilot generations (P2) ─> pilot labels ─> GATE 0
data splits (done) ───┘                                                               │
                                                                                      v
                                         full retrieval + generations + labels ─> GATE 1 (hand check, P4)
                                                                                      │
features (P3, needs only questions) ─────────────────────────────────────────────────>├─> probes (P3)
metrics + B1–B4, B8 (P4, synthetic labels first) ─────────────────────────────────────┘
```

P1 and P2 are the critical path, as Manya said. One addition: **P3's model check (D1) also blocks P2's full run**, because labels belong to one specific generator. So P3 starts on day 1 too.
P3 and P4 have useful work before any label exists. Features need only questions, and the metrics can be written and tested on fake labels.

## Week 0–1: Phase 0 pilot

**Person 1**
1. `uv sync --extra retrieval`; download the prebuilt index; check that `BM25Tool` returns sensible passages (confirm the title/text format).
2. Implement `retrieval/run.py`: top-100 per question, then holes or top-3, then gold.
3. Run it on `pilot`, then hand over `data/retrieval/{bm25,gold}/pilot.jsonl`.
4. Report hole statistics: how many passages were removed, and whether any hole question kept fewer than 3 passages.

**Person 2**
1. `uv sync --extra gpu` on the GPU node and serve Gemma 3 4B-IT with vLLM.
2. Implement `generation/run.py`. Run `closed` on the pilot immediately; it needs no retrieval.
3. Once P1's pilot retrieval lands, run `bm25` and `gold`; then `make_labels.py`.
4. Report class balance (gate: each class ≥ 15%), accuracy per condition, and generation time per 1k questions.

**Person 3**
1. Day 1–2: settle D1 (release, layers, widths, licence) and update the config.
2. Implement `features/extract.py` and run it on the pilot. Record extraction time.
3. Train the probe skeleton on pilot labels as soon as they exist (numbers won't mean anything yet; this tests the plumbing).

**Person 4**
1. Implement `eval/metrics.py` with unit tests on hand-made outcomes.
2. B1, B2, B3 (trivial once labels exist) and B8 (TF-IDF + LR on question text).
3. B4: run the verbalised router on the pilot (reuses P2's vLLM setup).
4. Write the hand-check protocol: a sheet of 200 sampled labels with the output, aliases and a verdict column.

**Gate 0:** each class ≥ 15% of pilot labels; the SAE is confirmed; measured compute replaces the proposal's estimates.
If ABSTAIN or RETRIEVE is under 15%, change `hole_fraction` or the dataset mix before the full run, not after.

## Weeks 1–2: full labels
P1 retrieves for all splits. P2 generates and labels all splits. P3 extracts features for all splits. P4 runs the hand check (gate: label error < 10%).

## Weeks 2–3: router and baselines
P3: probe grid (layers × widths × label budgets × seeds). P4: scores every router with the same metrics script.
Not yet assigned: B5 (FLARE/DRAGIN-style), B6 (SEAKR-style, 1k subset), B9 (QLoRA self-router). Assign them at the end of week 1, ideally to whoever is ahead.

## Weeks 3–4: transfer and latency
P1: dense retriever and labels for the shift test. P2: dense-condition generations. Everyone: latency at batch size 1 for their own stage.

## Weeks 4–5: ablations, failure analysis, write-up

## Hand-off rule
Results are passed as files, not code: `data/<stage>/...jsonl` keyed by `qid`, in the formats in `src/saerag/schema.py`. Every router writes `runs/routers/<id>/probs_<file>.jsonl`, so a single metrics script scores them all.
