# SAE-Gated RAG

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

## Person 3: probe MVP (CPU dry run, then GPU)

This answers a first question before the real labels exist: can a probe on the question's activation predict whether the model **knows** the answer?
Proxy label = closed-book answer correct (ANSWER) or not (NOT_ANSWER). Nothing is trained except logistic-regression probes; the LLM and the SAEs are frozen.

```bash
# 0. install: core + torch/transformers (a CPU-only torch is smaller)
uv sync --extra features
#    or: uv sync && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
#        && uv pip install transformers safetensors huggingface-hub scipy

# 1. questions + retrieval from Person 1's handoff (pilot.jsonl = 250 NQ + 250 TriviaQA).
#    Do not use another copy: every record is keyed to these qids and hole flags (D13).
mkdir -p data && tar xzf artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz -C data questions retrieval

# 2. CPU dry run on Gemma 3 1B-IT (ungated mirror) + Gemma Scope 2 1B SAEs (configs/cpu_1b.yaml)
#    one forward pass per question: activations at layers 7/13/17/22, 16k SAE codes,
#    and a greedy closed-book answer -> data/generations/closed_hf/pilot.jsonl
uv run python -m saerag.features.extract --config configs/cpu_1b.yaml --files pilot --limit 100 --closed-book
#    (or use Person 2's run: generation.run --condition closed, then --labels closed below)

# 3. probes vs baselines, 5-fold CV: majority, question-text TF-IDF, dense probe, SAE probe
uv run python -m saerag.probes.train --config configs/cpu_1b.yaml --train pilot --labels closed_hf \
    --layers 13 17 --budgets 50
```

What to check after step 2 (`data/features/gemma3-1b-it/stats_pilot.json`):
- `L*_l0` should be close to the SAE's target L0 (medium is about 30–60). If not, the hook layer is wrong.
- `L*_fvu` (fraction of variance unexplained) should be well below 1.
- The closed-book accuracy is printed. If almost nothing is correct, the 1B model knows too little and the proxy labels are lopsided.

Step 3 prints macro-F1 / AUROC / ECE per model and label budget. It writes `runs/probes/.../summary.csv`, top SAE features per class (look them up on Neuronpedia), and probe probabilities in `runs/routers/` for Person 4's metrics.

## Quick start

```bash
bash scripts/local_pilot.sh        # laptop: Gemma 3 1B on CPU, about 40 min, plumbing check only

# ADA: copy the code from your machine (the leading / keeps src/saerag/data/ in the copy)
rsync -av --exclude .venv --exclude /data --exclude /runs ./ <user>@ada.iiit.ac.in:saerag-proj/
# then on ADA, after accepting the Gemma 3 licence and `hf auth login`:
cd saerag-proj && bash scripts/ada_run.sh pilot     # Gate 0, then `full` for everything
```

Keep `DATA_ROOT` on storage the compute nodes can see, and check your quota first.

## Tonight: IIIT GPU cluster (Persons 2 + 3)

```bash
# login node (has internet)
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is missing
uv sync --extra features                                  # + `--extra gpu` for vLLM
hf auth login                                             # after accepting the Gemma 3 licence on HF
export SAERAG_DATA=/scratch/$USER/saerag/data HF_HOME=/scratch/$USER/hf
mkdir -p "$SAERAG_DATA" logs
tar xzf artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz -C "$SAERAG_DATA" questions retrieval
uv run python -m saerag.prefetch --widths 16k 262k        # Gemma 3 4B-IT + SAEs into HF_HOME
# edit `#SBATCH -A` (and partition if needed) in both scripts, then:
sbatch scripts/ada_tonight.sbatch                         # WIDE=1 also extracts 262k codes
sbatch --dependency=afterok:<job id> scripts/ada_grid.sbatch
```

If an older `$SAERAG_DATA` exists, delete its `generations/`, `labels/` and `features/` first: anything made from the old question set has different qids.

The job ([scripts/ada_tonight.sbatch](scripts/ada_tonight.sbatch)) runs, in order:
1. **P2:** closed-book, BM25 and gold generations for every split. Finished files are skipped, so resubmitting resumes after a timeout.
2. **P2:** labels (`make_labels`) and the Gate 0 report (`generation.report`, written to `runs/reports/`).
3. **P3:** activations at layers 4/9/13/17/22/29 and 16k SAE codes at 9/17/22/29 (`features.extract`).
4. **P3:** probes on the proxy label (`--labels closed`, i.e. ANSWER vs NOT_ANSWER):
   - 5-fold CV on the pilot
   - train on NQ+TriviaQA train, test on NQ / TriviaQA / PopQA / HotpotQA test, at budgets 50–1,000
5. **P3:** first real 3-way probes.

[scripts/ada_grid.sbatch](scripts/ada_grid.sbatch) is the CPU follow-up: the full probe grid for both label variants, the L1, binarised and mean-pooling ablations, and `probes.analyze` (top features, a hand-check sheet, drop-top-k).

Notes:
- **GPU and dtype.** `dtype: auto` uses bf16 on A100/3090/L4. On a 2080 Ti / T4 it uses fp32 split across GPUs: ask for 2 GPUs, or 3 for 262k SAEs. Never fp16, because Gemma 3 overflows. vLLM (`BACKEND=vllm`) is faster but wants a bf16 GPU; the default `hf` backend works anywhere.
- **262k SAEs.** About 2.7 GB per layer, encoder only. Run them as separate passes: `features.extract --files ... --sae-layers 17 --widths 262k`.
- **Out of memory?** Lower the batch size with `--batch-size 16`.
- **Logs.** Each generation run prints accuracy (plus accuracy on index-hole questions) and seconds per 1,000 questions. Extraction writes `stats_<file>.json` (SAE L0 / FVU). Probe results go to `runs/probes/.../summary.csv`.

**The real run (4B on a GPU).** Same commands without `--config configs/cpu_1b.yaml` and without `--limit`. Gemma 3 4B-IT is gated: accept the licence on Hugging Face, then run `hf auth login`.
- GPU: bf16 needs an L4, A100, 3090 or 4090 (Colab Pro is fine).
- Kaggle 2×T4 has no bf16. `dtype: auto` falls back to fp32 split across both GPUs.
- Once Person 2's labels exist, switch `--labels closed_hf` to `--labels default` and use `--train nq_train triviaqa_train --eval nq_test popqa_test`.

## Pipeline

| Step | Command | Output | Owner |
|------|---------|--------|-------|
| Splits | in Person 1's handoff; `uv run python -m saerag.data.build` reproduces them exactly | `data/questions/{dataset}_{split}.jsonl`, `pilot.jsonl` | done |
| Retrieval | `python -m saerag.retrieval.run --files pilot` | `data/retrieval/{bm25,gold}/` | P1 |
| Generation | `python -m saerag.generation.run --condition closed --files pilot` | `data/generations/{condition}/` | P2 |
| Labels | `python -m saerag.generation.make_labels --files pilot` | `data/labels/{variant}/` | P2 |
| Gate report | `python -m saerag.generation.report --files pilot` | `runs/reports/{tag}/` | P2 |
| Features | `python -m saerag.features.extract --files pilot` | `data/features/{tag}/L{layer}/` | P3 |
| Probes | `python -m saerag.probes.train` | `runs/routers/<id>/probs_*.jsonl` | P3 |
| Feature analysis | `python -m saerag.probes.analyze --layer 17 ...` | `runs/probes/.../analysis__*/` | P3 |
| Baselines, metrics | `saerag.baselines`, `saerag.eval` | `runs/` | P4 |

Shared, tested contract (change only after telling the team): `schema.py` (record formats), `answers.py` (correctness), `labels.py` (label rule), `prompts.py` (frozen at pilot).

## Git
Work on a branch per stream (`p1-retrieval`, `p2-labels`, `p3-probes`, `p4-eval`) and merge into `main` through small PRs. Never commit data, features or model files.
