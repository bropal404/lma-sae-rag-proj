# TODO

Move items from "To do" to "Done" as you finish them and commit the change. The current status and the plan behind each item are in the mid-evaluation report ([report/report_mid.pdf](report/report_mid.pdf)), sections 7 and 8.

**Status (2026-11-03):** BM25 retrieval is finished for all 13,500 questions, and Gemma 3 4B generations and labels exist for the benchmark splits. A two-way probe has run on 4B on a separate sample (test macro-F1: dense 0.71, SAE 0.67, question text 0.63, majority 0.40). The default label rule fails Gate 0, and the SAE probe scores lower with 2,000 labels than with 1,000. The rest of the project is now done by one person (Gopal).

## Done

### Shared groundwork
- [x] uv project, lockfile, tests (`uv run pytest`, 45 passing)
- [x] Question splits for all datasets + 500-question pilot, with index-hole flags
- [x] Shared contract: record formats, answer matching, label rule, prompts, config
- [x] ADA and Kaggle scripts, prefetch for the model and SAEs
- [x] 40-question end-to-end demo on Gemma 3 1B
- [x] Mid-evaluation report ([report/report_mid.pdf](report/report_mid.pdf))
- [x] All branches merged into `main`

### Person 1 (Nidhi): retrieval
- [x] Search tool, BM25 search, hole filter and gold stand-in
- [x] Lucene index over the full 2018 Wikipedia passages (21,015,324 passages)
- [x] Full retrieval for all splits, handed over in `artifacts/retrieval/` and validated (no lexical hole leaks)

### Person 2 (Manya): generation and labels
- [x] Generation (vLLM or HF backend), labels and gate report
- [x] Gemma 3 4B closed-book, BM25 and gold generations and labels on the benchmark splits (Kaggle, 2×T4; ~140 s per 1,000 questions closed-book, ~1,250 s with BM25)
- [x] Gate 0 check: default rule fails (RETRIEVE 11.2%); `miss_as_retrieve` passes on the pilot (35 / 36 / 28) but ABSTAIN is 13% on TriviaQA

### Person 3 (Gopal): activations, SAEs and probes
- [x] Feature extraction, probe training and analysis code
- [x] Activations and SAE codes on a sample; two-way probes on 4B

### Person 4 (Salvik): baselines and metrics
- [x] B1–B4 and B8, metrics with unit tests, one scoring script for every router and probe
- [x] 200-label hand-check sheet and protocol ([docs/HANDCHECK.md](docs/HANDCHECK.md))

## To do

From here on one person does everything, so this is a single ordered list. Items under "Core" answer the hypotheses; do them in order. Items under "If time allows" are cut first.

### Core
1. [ ] Choose the label rule (default or `miss_as_retrieve`), then remake labels for all splits. Everything below depends on it.
2. [ ] Find the cause of the SAE probe scoring lower with 2,000 labels than with 1,000.
3. [ ] Extract activations and SAE codes on the canonical splits: dense at layers 4, 9, 13, 17, 22, 29; SAEs at 9, 17, 22, 29, width 16k.
4. [ ] Three-way probes on the 4B labels, SAE and dense, then the grid: layers × label budgets (50 to 5,000) × seeds (H2).
5. [ ] Run B1–B3 and B8 on the real labels, and B4 on GPU.
6. [ ] Score every router and probe with `eval/score.py`, split by all / hole / non-hole / dataset (Gate 2, H1, H3).
7. [ ] Latency at batch size 1: BM25 per question, generation per condition, controller (forward pass + probe); p50 and p95 (Gate 3, H4).
8. [ ] Hand check (Gate 1: label error < 10%). If 200 labels is too many alone, check 100 and report the interval.
9. [ ] Final report.

### If time allows
- [ ] 262k-width SAEs at the best layer
- [ ] Ablations with `scripts/ada_grid.sbatch`: L1, binarised, mean-pooling
- [ ] Feature analysis: top features per class, 30 examples by hand, drop top-k and measure the fall in F1
- [ ] Review the sampled hole passages (`docs/retrieval/hole_review_samples.json`)
- [ ] Failure analysis, including PopQA popularity bins
- [ ] Dense retriever (e5-base-v2, 2M passages, FAISS) and its generation condition, for the retriever-shift part of H2
- [ ] B6 sampled-uncertainty trigger

### Open decisions
- [ ] Gold condition for NQ/TriviaQA: keep the answer-bearing BM25 stand-in (372 of 1,000 PopQA questions have none), or move to DPR's gold files
- [ ] The BM25 index is on ADA under Nidhi's home directory; copy it if access may lapse

### Gates
- [ ] Gate 0: each class ≥ 15% of pilot labels (default fails; variant passes)
- [ ] Gate 1: label error < 10% on the hand check
- [ ] Gate 2: a router cuts retrieval calls versus B1 with at most 2 points of accuracy loss
- [ ] Gate 3: latency measured for every method

### Dropped
- B5 generation-time trigger, B9 QLoRA self-router, B10 post-retrieval refusal
