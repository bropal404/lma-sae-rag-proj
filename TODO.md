# TODO

Tick items as you finish them and commit the change. The details behind each item are in [docs/PLAN.md](docs/PLAN.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

## Done (shared groundwork)
- [x] uv project, git, lockfile, tests (`uv run pytest`)
- [x] Question splits for all datasets + 500-question pilot, with index-hole flags (`uv run python -m saerag.data.build`)
- [x] Shared contract: record formats (`schema.py`), answer matching (`answers.py`), label rule (`labels.py`), prompts (`prompts.py`), config
- [x] Stubs with fixed inputs/outputs for every work stream
- [x] One question set: Person 1's build is canonical (D13); `src/saerag/data/` is now tracked

## Team (this week)
- [ ] Fill in owner names in `docs/PLAN.md`
- [ ] H3 framing: the probe cannot see index holes (D3 caveat). Agree to report hole and non-hole questions separately?
- [ ] Gold condition for NQ/TriviaQA: answer-bearing BM25 passage as a stand-in, or DPR's gold files (D10)
- [ ] Shared storage for `data/` and the Pyserini cache (D11)
- [ ] Assign B5, B6 and B9 at the end of week 1

## Person 1: retrieval
- [x] Search tool interface (`retrieval/tool.py`)
- [x] BM25 search, first version (`retrieval/bm25.py`, not yet run)
- [x] Index-hole filter and gold stand-in (`retrieval/holes.py`, tested)
- [ ] `uv sync --extra retrieval`; download the prebuilt index (~10 GB); check the passage title/text format in `bm25.py`
- [ ] Write `retrieval/run.py`: top-100 per question, then hole filter or top-3, plus gold passages and latency
- [ ] Run on the pilot; hand over `data/retrieval/{bm25,gold}/pilot.jsonl`
- [ ] Report hole statistics (passages removed, hole questions left with fewer than 3 passages)
- [ ] Full retrieval for all splits
- [ ] Week 3: dense retriever (e5-base-v2 over a 2M-passage subsample, FAISS) for the retriever-shift test

## Person 2: generation and labels
- [x] Answer matching, label rule and variant, prompts (tested)
- [ ] `uv sync --extra gpu` on the GPU node; get Gemma 3 4B-IT running in vLLM
- [x] Write `generation/run.py` (vLLM or HF backend; not yet run)
- [ ] Run the closed-book condition on the pilot (needs no retrieval)
- [ ] Run the BM25 and gold conditions on the pilot once Person 1's retrieval is ready
- [x] Write `generation/make_labels.py` (not yet run)
- [x] Write `generation/report.py` (accuracy, class balance, time, gate; tested)
- [ ] Report pilot class balance (gate: each class ≥ 15%), accuracy per condition, time per 1,000 questions
- [ ] Full generations and labels for all splits (waits on Person 3 confirming the model)
- [ ] Week 3: dense-retriever condition

## Person 3: activations, SAEs and probes
- [x] **Day 1–2, blocks everyone:** confirm Gemma Scope 2 covers Gemma 3 4B-IT (IT-model resid_post SAEs at layers 9/17/22/29, widths 16k–1M, CC-BY-4.0)
- [x] Fill in the SAE settings in `configs/default.yaml`
- [ ] Accept the Gemma 3 licence on HF (the model is gated)
- [x] Write `features/extract.py` (last-token and mean-pooled activations, SAE codes; not yet run)
- [ ] Run it on the pilot and record the time
- [x] Write `probes/train.py` (logistic regression with L1 or L2, class-weighted, CV or held-out, label budgets; not yet run)
- [ ] MVP: probes on the proxy label (closed-book correct) via `scripts/ada_tonight.sbatch`
- [ ] Extract features for all splits
- [x] Write the grid job (`scripts/ada_grid.sbatch`) and `probes/analyze.py` (tested on synthetic data)
- [ ] Probe grid: layers × widths × label budgets (50 to 5,000) × seeds, SAE and dense (B7)
- [ ] Feature analysis: top features per class, check 30 examples by hand, drop top-k features and measure the fall in F1

## Person 4: baselines, metrics and hand check
- [x] B1 always retrieve, B2 never retrieve, B3 oracle (`baselines/routers.py`)
- [x] Routing outcome helper (`eval/metrics.py`)
- [ ] Metrics with unit tests on hand-made data:
  - [ ] accuracy and selective accuracy
  - [ ] hallucination rate (index-hole questions reported separately)
  - [ ] retrieval-call rate, and precision/recall of the RETRIEVE decision
  - [ ] abstention recall at 10% false refusal (threshold tuned on val)
  - [ ] risk-coverage curve and AURC
  - [ ] macro-F1 and calibration error
  - [ ] latency p50 and p95, split by stage
- [ ] B4 verbalised router on the pilot (reuses Person 2's vLLM setup)
- [ ] B8 question-text classifier (TF-IDF + logistic regression first)
- [ ] 200-label hand-check sheet and protocol (watch the short PopQA aliases like "AR" and "PL")
- [ ] Run the hand check (gate: label error < 10%)

## Gates
- [ ] Gate 0 (end of week 1): each class ≥ 15% of pilot labels, SAE confirmed, compute re-measured
- [ ] Gate 1 (week 2): label error < 10% on the hand check
- [ ] Gate 2 (week 3): a router cuts retrieval calls versus B1 with at most 2 points of accuracy loss
- [ ] Gate 3 (week 4): latency measured for every method

## Not assigned yet
- [ ] B5 generation-time trigger (FLARE/DRAGIN-style)
- [ ] B6 sampled-uncertainty trigger (SEAKR-style, 1,000-question subset)
- [ ] B9 QLoRA self-router
- [ ] B10 post-retrieval refusal (optional)
- [ ] Ablations, failure analysis, write-up (weeks 4–5)
