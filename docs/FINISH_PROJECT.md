# Finishing SAE-Gated RAG from the November 11 AWS snapshot

This runbook uses the **newer in-repository TODO.md**, not the older four-person TODO shared separately. The core training and GPU experiments are already reflected in `results/sagemaker/` and the end of `logs/run.log`. Do **not** run `scripts/sagemaker/run.sh` again just because the TODO checkboxes remain unchecked: it can reinstall packages, repeat expensive GPU work, overwrite outputs and push commits.

## Decisions and reproducible configuration

- Primary label rule: **`miss_as_retrieve`**; the default rule fails Gate 0.
- Main comparisons: **validation-selected** full-budget SAE 16k L29 vs dense L22. The L17 comparison remains relevant for historical latency and feature analyses, but do not call it the validation-selected primary controller.
- Training: NQ train + TriviaQA train. Layer selection: NQ validation + TriviaQA validation only. Test: NQ, TriviaQA, PopQA, HotpotQA, plus SelfAware abstention.
- Original output exports in `results/sagemaker/` are a small subset. The attached ZIP **does not contain** the canonical `data/labels`, `data/features`, or `runs/routers` outputs. Find the SageMaker workspace holding those data before recomputing statistics.
- Strict handcheck: 16/200 label errors (8%); lenient: 22/200 (11%). The 95% Wilson interval for strict is approximately 5.0–12.6%. Decide which protocol is authoritative and report both as sensitivity analyses. Do not redo the human work unless the existing sheet is genuinely invalid.

## A. Apply the patch safely

1. On the Git checkout or SageMaker workspace, confirm branch and clean tree:
   `git status -sb`
2. Start a new branch: `git switch -c final-evaluation` (if it does not already exist).
3. Back up any *uncommitted* changes before applying the patch. Place `saerag_finish_patch.zip` at the repo root and extract it there: `unzip -o saerag_finish_patch.zip`.
4. Run the CPU tests: `PYTHONPATH=src .venv/bin/python -m pytest -q` (or `uv run pytest -q` if the environment was created with uv).
5. Generate a report immediately from the **exported** results: `.venv/bin/python scripts/finalize_project.py`. This writes `results/final/`. No model or raw data is needed for this audit.

## B. Confirm the complete raw files (SageMaker, CPU)

**Before this step:** work on the machine with `data/labels/`, `data/features/` and `runs/routers/`. If these are missing, locate the pre-existing `results_bundle.tar.gz` or connected storage; do not start generation again.

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_raw_outputs.py --config configs/sagemaker.yaml --variant miss_as_retrieve --layer 29 --check-probs
```

Expected: 14,000 validated question/label pairs counting the 500 overlapping pilot records. Stops on missing rows, inconsistent label rule, mismatched QIDs, or incomplete features. If your canonical files live under a nondefault root, export `SAERAG_DATA` before running.

## C. Fix the bootstrap confidence intervals (SageMaker, CPU)

The **previous** `results/analysis/sae_vs_dense_bootstrap.csv` was produced by code that sampled QIDs with replacement and then created dictionaries. This silently removed duplicates and invalidated the confidence intervals.

```bash
PYTHONPATH=src .venv/bin/python scripts/analysis/bundle_analysis.py
```

This version resamples row positions with replacement and recomputes paired differences using all sampled observations. The script also recreates confusion, PopQA popularity and routing-error CSVs. It needs canonical test labels, generations and all three-layer router probabilities. Keep the old CSV in Git history; don't cite its CI after fixing the analysis.

## D. Measure the primary L29 controller and routing latency (SageMaker GPU)

First verify that `runs/latency/bm25.jsonl` exists; otherwise H4 would omit retrieval cost. The existing BM25 timing uses batch size 1 on a matched 200-question subset (100 NQ + 100 TriviaQA).

```bash
test -s runs/latency/bm25.jsonl
PYTHONPATH=src .venv/bin/python scripts/sagemaker/latency_measure.py --config configs/sagemaker.yaml --layer 29 --variant miss_as_retrieve --out-name l29
```

The `--out-name l29` option preserves the previous L17 timing files. Once that GPU step completes:

```bash
PYTHONPATH=src .venv/bin/python scripts/sagemaker/latency_route.py --config configs/sagemaker.yaml --variant miss_as_retrieve --layer 29 --stage-file stages_l29.jsonl --out routed_l29.csv --match-scoring
```

`--match-scoring` uses the same validation-tuned abstention threshold as `saerag.eval.score`; the previous latency route script used argmax and therefore timed a **different operating point** from the accuracy results.

Export the new latency summaries for inclusion in Git:

```bash
cp runs/latency/routed_l29.csv results/sagemaker/latency/
cp runs/latency/stages_l29_summary.csv results/sagemaker/latency/
```

## E. Regenerate the final audit (CPU)

```bash
PYTHONPATH=src .venv/bin/python scripts/finalize_project.py
```

Check these generated files:
- `results/final/final_summary.md` — Gate 0–3, H1–H4 with limitations
- `results/final/gate_summary.json` — machine-readable gate evidence
- `results/final/gate2_by_dataset.csv`, `gate2_aggregate.csv` — per-dataset and weighted 4,000-answerable-question evaluation
- `results/final/model_selection.csv` — validation-only SAE/dense selection
- `results/final/h2_label_efficiency.csv` — label-budget sweep
- `results/final/h3_hole_risk.csv` — three-way vs two-way risk on hole questions
- `results/final/report_draft.md` — academic report skeleton with actual values; review and polish against the mid-evaluation paper

**Interpretation** (snapshot before new latency): Gate 0 passes on the exported pilot, strict Gate 1 passes by point estimate, and the validation-selected SAE achieves Gate 2/H1 with about 51% fewer retrieval calls and about 0.35 points lower accuracy than B1 across answerable test data. H3 has lower *wrong answers per hole question* than the two-way model, with lower coverage due to abstaining. H2's validation-selected SAE trails validation-selected dense by about 2.38 test macro-F1 points, slightly outside the stated two-point margin. Do not claim universal SAE superiority. H4 for selected L29 needs step D.

## F. Complete the academic report and push

1. Start with `results/final/report_draft.md`, carry over verified background, formulas, plots and reference entries from `report/report_mid.typ`, and revise the text to reflect **full 4B three-way experiments** rather than the earlier 1B pilot/two-way results. Remove mid-evaluation claims that were superseded.
2. Report both strict and lenient human judgements, the Gate 1 uncertainty, selected-on-validation protocol, exactly which test sets were aggregated, 95% CIs recomputed in C, and L29 latency in D.
3. Explain limitations: query-time **lexical** holes, BM25-derived gold proxies (including 372 PopQA cases without one), output string-matching error, and SelfAware unanswerable questions needing abstention recall rather than answer accuracy.
4. Treat the dense retriever shift and B6 uncertainty router as **optional** per the newer TODO. They require substantial new indexing/generation work and are not prerequisites to the existing core results. If not run, say so and don't claim retriever-shift performance.
5. Review `git diff --check`, tests, generated figures/CSV and final report. Update TODO checkboxes *only for verified work* and commit on your feature branch.

## Minimum quality gates before calling it complete

- Gate 0: all three pilot label classes at least 15% **on current canonical labels** (not merely a possibly older report export).
- Gate 1: report 16/200 strict error, CI and alternative adjudication. Point estimate alone meets the defined cutoff; the CI still overlaps 10%.
- Gate 2: accuracy no worse than B1 minus 0.02 and ≥30% fewer retrieval calls on the **combined answerable test questions**, and show every dataset separately.
- Gate 3: batch-size-1 p50 and p95 of controller, retrieval, closed and RAG generation, B4 and selected routing policy, measured on the **same sample and same thresholded routing decisions**.
- H2: state whether within-two-point comparison holds on test; explain any negative result, don't adjust the model after looking at test.
- H3: report conditional wrong-answer rate, unconditional unsafe-answer rate and coverage on hole questions.
- Recalculated paired-bootstrap CIs are used in the final report.
- Reproducible checkpoint/seed/corpus details and correct Git commit are recorded.
