# TODO

Move items from "To do" to "Done" as you finish them and commit the change. The current status and the plan behind each item are in the mid-evaluation report ([report/report_mid.pdf](report/report_mid.pdf)), sections 7 and 8.

**Status (2026-11-03):** BM25 retrieval is finished for all 13,500 questions, and Gemma 3 4B generations and labels exist for the benchmark splits. A two-way probe has run on 4B on a separate sample (test macro-F1: dense 0.71, SAE 0.67, question text 0.63, majority 0.40). The default label rule fails Gate 0, and the SAE probe scores lower with 2,000 labels than with 1,000. Day counts below are rough estimates; each person has about 4.5–5 days of core work.

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

### Nidhi (~5 days)
- [ ] Dense retriever: e5-base-v2 over a 2M-passage subsample, FAISS index; hand over to Manya *(1 day, week 1)*
- [ ] BM25 latency per question at batch size 1, for Gate 3 *(0.5 day, week 1)*
- [ ] Review the sampled hole passages by hand (`docs/retrieval/hole_review_samples.json`) *(0.5 day, week 1)*
- [ ] B6 sampled-uncertainty trigger (SEAKR-style, 1,000-question subset) *(2 days, week 2)*
- [ ] Feature analysis after Gopal's probes: top features per class, check 30 examples by hand, drop top-k features and measure the fall in F1 *(1 day, week 3)*
- [ ] Report sections: retrieval, holes, retriever shift, SAE features

### Manya (~5 days)
- [ ] **First:** settle the label rule (default or `miss_as_retrieve`) with the team, then remake labels for all splits; this blocks Gopal and Salvik *(1 day, week 1)*
- [ ] Generation latency at batch size 1 for each condition *(0.5 day, week 1)*
- [ ] Dense-retriever generation condition, after Nidhi's index (~5 h compute) *(1 day, week 2)*
- [ ] Grid ablations with `scripts/ada_grid.sbatch`: L1, binarised, mean-pooling, after Gopal's extraction *(1 day, week 2)*
- [ ] Failure analysis, including PopQA popularity bins *(1.5 days, week 3)*
- [ ] Report sections: data, label rule, Gate 0

### Gopal (~4.5 days, critical path)
- [ ] **Start now:** find the cause of the 1,000 vs 2,000 label SAE result *(1–2 days, week 1)*
- [ ] Extract activations and SAE codes on the canonical splits after the new labels: dense at layers 4, 9, 13, 17, 22, 29; SAEs at 9, 17, 22, 29, widths 16k and 262k *(1 day, week 1)*
- [ ] Three-way probes on the 4B labels, then the main grid: layers × widths × label budgets (50 to 5,000) × seeds, SAE and dense (B7) *(1.5 days, week 2)*
- [ ] Report sections: method, H2 (SAE vs dense), label efficiency
- [ ] If the debug runs past 2 days, say so early so Manya can start the extraction

### Salvik (~5 days)
- [ ] **Now:** the 200-label hand check (Gate 1: label error < 10%); recheck after the label rule changes *(1 day, week 1)*
- [ ] Run B4 on GPU, and B1–B3 and B8 on the real labels *(1 day, week 1)*
- [ ] Controller latency (forward pass + probe) at batch size 1, and the full H4 latency table from everyone's timings *(1 day, week 2)*
- [ ] Score every router and probe with `eval/score.py`, split by all / hole / non-hole / dataset (Gate 2, H1, H3) *(1.5 days, week 2–3)*
- [ ] Report sections: results tables and figures, limitations

### Team
- [ ] Agree on the label rule (week 1, led by Manya)
- [ ] Gold condition for NQ/TriviaQA: keep the answer-bearing BM25 stand-in, or move to DPR's gold files? 372 of 1,000 PopQA questions have no stand-in
- [ ] Shared storage for `data/` (the index is on ADA under Nidhi's home directory)
- [ ] Final report: everyone edits the full draft together (week 3)

### Gates
- [ ] Gate 0: each class ≥ 15% of pilot labels (default fails; variant passes)
- [ ] Gate 1: label error < 10% on the hand check
- [ ] Gate 2: a router cuts retrieval calls versus B1 with at most 2 points of accuracy loss
- [ ] Gate 3: latency measured for every method

### Dropped unless someone finishes early
- [ ] B5 generation-time trigger (FLARE/DRAGIN-style)
- [ ] B9 QLoRA self-router
- [ ] B10 post-retrieval refusal
