# SAE-Gated RAG — Project Status Report

*Snapshot of all team branches as of 14 Nov 2026. Numbers come from `origin/final-evaluation` (latest branch, includes the `aws` SageMaker runs).*

**Goal.** Before generating, decide for each question whether the model should **ANSWER** from memory, **RETRIEVE** with BM25, or **ABSTAIN**. The router is a small probe on frozen Gemma 3 4B activations, built on either Gemma Scope 2 SAE features or raw (dense) residual activations.

---

## 1. Headline results

| Check | Result | Evidence |
|---|---|---|
| **Gate 0**: every label class ≥ 15% | ✅ PASS | ANSWER 35.4% · RETRIEVE 36.6% · ABSTAIN 28.0% |
| **Gate 1**: label error < 10% (hand check) | ✅ PASS | 15/200 = **7.5%** (Wilson 95% CI 4.6–12.0%); string-match error 4.8% |
| **Gate 2 / H1**: ≤ 2 pt accuracy loss, ≥ 30% fewer retrievals | ✅ PASS | SAE router 31.5% vs always-retrieve 31.9% (−0.35 pt) with **50.8% fewer BM25 calls** |
| **H3**: three-way routing cuts unsafe answers on index holes | ✅ Supported | Wrong answers per hole question **61.0%** (3-way) vs 82.2% (2-way) |
| **Gate 3 / H4**: latency at batch size 1 | ✅ Measured (L17 router) | Controller **56.5 ms** p50; SAE-routed pipeline 212 ms p50 vs 251 ms for always-retrieve |
| **H2**: SAE vs dense probe | Comparable | Dense L22 is ahead by 1.2 pt (val) / 2.4 pt (test) macro-F1 |

---

## 2. Work completed

### Data and retrieval (Person 1)
- Lucene BM25 index over the full **2018 Wikipedia (21,015,324 passages)**.
- Retrieval done for all **13,500 questions** in NQ, TriviaQA, PopQA, HotpotQA and SelfAware, plus a 500-question pilot.
- Index-hole simulation (answer-bearing passages filtered from the top 100), gold stand-in passages, and a validated handoff archive with no lexical hole leaks.

### Generation and labels (Person 2)
- Gemma 3 4B **closed-book, BM25 and gold** generations on every split, with vLLM and HF backends (Kaggle 2×T4 and SageMaker L4).
- Label rule settled on **`miss_as_retrieve`** (the default rule failed Gate 0). Answer matcher fixed, generations rescored, labels rebuilt.
- Generation cost measured and a failure analysis written.

### Activations, SAEs and probes (Person 3)
- Dense activations at layers 9/17/22/29 and SAE codes (16k width) at layers 9/17/22/29; **262k-width SAEs** at L17/L22.
- Three-way and two-way L2 probes on SAE, dense, TF-IDF and majority features.
- **Label-budget grid** from 50 to 5,000 labels plus the full set, 3 seeds each (H2 label efficiency).
- **Ablations**: L1 penalty, binarised features, mean pooling, C-grid, extra layers.
- Layer selection on validation only: **SAE 16k @ L29** and **dense @ L22**.
- The earlier 1k→2k-label drop does not reappear with more seeds (F1 0.533 → 0.544 → 0.563 at 1k/2k/4k).

### Baselines, metrics and evaluation (Person 4)
- Baselines **B1** always-retrieve, **B2** never-retrieve, **B3** oracle, **B4** verbalised self-router (GPU), **B8** TF-IDF + LR.
- Metrics module with unit tests, and **one scoring script** (`eval/score.py`) for every router and probe, split by all / hole / non-hole / dataset.
- **Hand-check tool and protocol** (`docs/HANDCHECK.md`). 200 labels hand-checked, with strict and lenient adjudication.
- Paired bootstrap CIs (resampling bug fixed), confusion matrices, routing-error breakdown, PopQA popularity bins.

### Interpretability
- Top SAE features per class looked up on **Neuronpedia**.
- **Drop-top-k test** (L22): zeroing the top 10 features per class (30 total) lowers macro-F1 from 0.535 to 0.445; retraining without them recovers 0.510. Routing information is concentrated in a small set of features.

### Infrastructure
- `uv` project with tests, ADA/Slurm and SageMaker run scripts, Kaggle notebooks, prefetch for the model and SAEs.
- Mid-evaluation report (`report/report_mid.pdf`, 14 figures).
- Final audit script (`scripts/finalize_project.py`) that writes `results/final/` (gate summary, per-dataset tables, H2/H3 tables, report draft).

---

## 3. Detailed numbers

### Gate 2 / H1: answerable test questions (n = 4,000)

| Router | Accuracy | BM25 calls | Δ vs B1 |
|---|---:|---:|---:|
| B1 always-retrieve | 31.9% | 100% | — |
| B2 never-retrieve | 26.4% | 0% | −5.5 pt |
| B4 verbalised | 31.3% | 64.7% | −0.6 pt |
| B8 TF-IDF + LR | 27.8% | 38.0% | −4.1 pt |
| **SAE 16k @ L29** | **31.5%** | **49.2%** | **−0.35 pt** |
| Dense @ L22 | 34.0% | 60.4% | +2.1 pt |

Per dataset, the SAE router beats B1 on NQ (22.3 vs 20.0) and TriviaQA (53.0 vs 51.0) with about half or fewer of the retrievals, and roughly matches it on HotpotQA (22.2 vs 22.5). PopQA (long-tail entities) relies most on retrieval.

### H2: SAE vs dense (validation-selected)
- Validation macro-F1: SAE 51.8%, dense 53.1%.
- Test (mean of 4 datasets): SAE 45.8%, dense 48.2%. Paired bootstrap on the routed decisions: −1.9 pt [−3.5, −0.3].

### H3: index-hole questions (NQ + TriviaQA + HotpotQA, n = 894)

| | Wrong per hole question | Coverage | Wrong among attempted |
|---|---:|---:|---:|
| Three-way SAE | **61.0%** | 84.3% | 72.3% |
| Two-way SAE | 82.2% | 100% | 82.2% |

### Latency: batch size 1, NVIDIA L4, n = 200

| Stage / router | p50 | p95 |
|---|---:|---:|
| Controller (forward pass + probe) | 56.5 ms | 58.5 ms |
| BM25 top-3 | 19.1 ms | 108.3 ms |
| Closed-book generation | 201 ms | 400 ms |
| RAG generation | 230 ms | 513 ms |
| B1 always-retrieve (end-to-end) | 251 ms | 541 ms |
| B4 verbalised | 466 ms | 790 ms |
| B8 TF-IDF | 155 ms | 411 ms |
| SAE router (L17) | 212 ms | 480 ms |

---

## 4. Next steps

- Run the selected-layer (SAE L29 / dense L22) latency benchmark on SageMaker. The scripts were added on 14 Nov (`docs/SELECTED_LATENCY_SAGEMAKER.md`); current latency figures use the L17 router.
- Turn `results/final/report_draft.md` into the final paper, starting from the mid-eval Typst source.
- Merge Manya's parallel Kaggle branch and tick off the finished items in `TODO.md`.
- Optional extensions: dense retriever (e5) for the retriever-shift part of H2, and the B6 sampled-uncertainty baseline.
