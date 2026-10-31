# TODO

Move items from "To do" to "Done" as you finish them and commit the change. The details behind each item are in [docs/PLAN.md](docs/PLAN.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

**Status (2026-10-30):** the code for retrieval, generation, labels, features and probes is written, and full BM25 retrieval is finished. No generation, feature or probe job has been recorded as run with Gemma 3 4B yet, so there are no labels and no gate has been passed. The next step is the ADA pilot (Gate 0).

## Done

### Shared groundwork
- [x] uv project, git, lockfile, tests (`uv run pytest`, 28 passing)
- [x] Question splits for all datasets + 500-question pilot, with index-hole flags
- [x] Shared contract: record formats (`schema.py`), answer matching (`answers.py`), label rule (`labels.py`), prompts (`prompts.py`), config
- [x] One question set: Person 1's build is canonical (D13); `src/saerag/data/` is now tracked and reproduces the handoff byte for byte
- [x] ADA scripts: `scripts/ada_run.sh` (`pilot` or `full`), `ada_tonight.sbatch`, `ada_grid.sbatch`, and `saerag.prefetch` for the model and SAEs
- [x] Laptop plumbing check written: `scripts/local_pilot.sh` (Gemma 3 1B on CPU, not yet run)

### Person 1: retrieval
- [x] Search tool interface (`retrieval/tool.py`), BM25 search (`retrieval/bm25.py`), hole filter and gold stand-in (`retrieval/holes.py`)
- [x] Lucene index over the full 2018 Wikipedia passages (21,015,324 passages), kept at `/home2/nidhiv/saerag-person1/wiki-full/index` on ADA
- [x] `retrieval/run.py`: top-100 per question, then hole filter or top-3, plus gold passages
- [x] Pilot run and handover (`data/retrieval/{bm25,gold}/pilot.jsonl`)
- [x] Hole statistics on the pilot: 157 of 500 hole questions, 1,800 passages removed, none left with fewer than 3 passages
- [x] Full retrieval for all splits (13,500 questions), handed over in `artifacts/retrieval/`, validated by `scripts/retrieval/validate_person1_handoff.py` (no lexical hole leaks)

### Person 2: generation and labels
- [x] Answer matching, label rule and variant, prompts (tested)
- [x] `generation/run.py` (vLLM or HF backend; not yet run on 4B)
- [x] `generation/make_labels.py` (not yet run on 4B)
- [x] `generation/report.py` (accuracy, class balance, time, gate; tested)

### Person 3: activations, SAEs and probes
- [x] Gemma Scope 2 covers Gemma 3 4B-IT (IT-model resid_post SAEs at layers 9/17/22/29, widths 16k–1M, CC-BY-4.0); settings are in `configs/default.yaml`
- [x] `features/extract.py` (last-token and mean-pooled activations, SAE codes; not yet run on 4B)
- [x] `probes/train.py` (logistic regression with L1 or L2, class-weighted, CV or held-out, label budgets; majority, quick TF-IDF, dense B7 and SAE probes; not yet run on 4B)
- [x] `scripts/ada_grid.sbatch` (probe grid plus L1, binarised and mean-pooling ablations)
- [x] `probes/analyze.py` (tested on synthetic data)

### Person 4: baselines and metrics
- [x] B1 always retrieve, B2 never retrieve, B3 oracle (`baselines/routers.py`)
- [x] Routing outcome helper (`eval/metrics.py`)

## To do

### Next, in order
1. [ ] Commit `scripts/ada_run.sh`, `scripts/local_pilot.sh` and the README and sbatch edits
2. [ ] Accept the Gemma 3 licence on HF and run `hf auth login` on ADA (the model is gated)
3. [ ] Copy the code to ADA and run `bash scripts/ada_run.sh pilot`
4. [ ] Read the Gate 0 report; if a class is under 15%, change `hole_fraction` or the dataset mix before the full run
5. [ ] Run `bash scripts/ada_run.sh full`

### Team
- [ ] Fill in owner names in `docs/PLAN.md`
- [ ] H3 framing: the probe cannot see index holes (D3 caveat). Agree to report hole and non-hole questions separately?
- [ ] Gold condition for NQ/TriviaQA (D10): the handoff uses the answer-bearing BM25 passage as a stand-in. Keep it, or move to DPR's gold files? Note that 372 of 1,000 PopQA questions have no stand-in
- [ ] Shared storage for `data/` (D11). The index is on ADA under Person 1's home directory, and nothing else is agreed
- [ ] Assign B5, B6 and B9
- [ ] Merge into `main`: it still has only the scaffold, and all the work above is on `nidhi` and `gopal`
- [ ] Fix the README's "Tonight" section: it says to edit `#SBATCH -A`, but that line is gone from the scripts (`ACCOUNT=` in `ada_run.sh` replaces it)

### Person 1: retrieval
- [ ] Review the sampled hole passages by hand (`docs/retrieval/hole_review_samples.json`) before any final claim
- [ ] Measure per-question BM25 latency: the batch run leaves it unset on purpose, and Gate 3 needs it
- [ ] Week 3: dense retriever (e5-base-v2 over a 2M-passage subsample, FAISS) for the retriever-shift test

### Person 2: generation and labels
- [ ] Get Gemma 3 4B-IT running on ADA (`hf` backend by default; vLLM needs a bf16 GPU)
- [ ] Run the closed-book, BM25 and gold conditions on the pilot
- [ ] Report pilot class balance (gate: each class ≥ 15%), accuracy per condition, time per 1,000 questions
- [ ] Full generations and labels for all splits
- [ ] Week 3: dense-retriever condition

### Person 3: activations, SAEs and probes
- [ ] Run `features/extract.py` on the pilot and record the time; check SAE L0 and FVU in `stats_pilot.json`
- [ ] MVP: probes on the proxy label (closed-book correct)
- [ ] Extract features for all splits
- [ ] Probe grid: layers × widths × label budgets (50 to 5,000) × seeds, SAE and dense (B7)
- [ ] Feature analysis: top features per class, check 30 examples by hand, drop top-k features and measure the fall in F1

### Person 4: baselines, metrics and hand check
- [ ] Metrics with unit tests on hand-made data (`eval/metrics.py`):
  - [ ] accuracy and selective accuracy
  - [ ] hallucination rate (index-hole questions reported separately)
  - [ ] retrieval-call rate, and precision/recall of the RETRIEVE decision
  - [ ] abstention recall at 10% false refusal (threshold tuned on val)
  - [ ] risk-coverage curve and AURC
  - [ ] macro-F1 and calibration error
  - [ ] latency p50 and p95, split by stage
- [ ] B4 verbalised router on the pilot (reuses Person 2's setup)
- [ ] B8 question-text classifier: full version (`probes/train.py` has only a quick TF-IDF one)
- [ ] 200-label hand-check sheet and protocol (watch the short PopQA aliases like "AR" and "PL")
- [ ] Run the hand check (gate: label error < 10%)

### Gates
- [ ] Gate 0 (end of week 1): each class ≥ 15% of pilot labels, compute re-measured (SAE already confirmed)
- [ ] Gate 1 (week 2): label error < 10% on the hand check
- [ ] Gate 2 (week 3): a router cuts retrieval calls versus B1 with at most 2 points of accuracy loss
- [ ] Gate 3 (week 4): latency measured for every method

### Not assigned yet
- [ ] B5 generation-time trigger (FLARE/DRAGIN-style)
- [ ] B6 sampled-uncertainty trigger (SEAKR-style, 1,000-question subset)
- [ ] B9 QLoRA self-router
- [ ] B10 post-retrieval refusal (optional)
- [ ] Ablations beyond the grid job, failure analysis, write-up (weeks 4–5)
