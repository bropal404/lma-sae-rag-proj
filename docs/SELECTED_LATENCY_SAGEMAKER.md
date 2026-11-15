# Final Gate 3/H4: SAE L29 versus dense L22 on SageMaker

These files are additions, not replacements for the original L17 scripts.
They require the **original persistent SageMaker workspace** and its feature arrays.
They are not intended to run on the ADA login node.

## Why separate scripts?

The earlier `latency_measure.py --layer 29` measured an SAE L29 controller
and then reused its controller cost for dense L29. That would not accurately
time the locked validation-selected SAE L29 versus dense L22 comparison.

`latency_selected_measure.py` re-fits the original L2 probe architecture and
train-only inner CV on existing NQ/TriviaQA training arrays (no new feature
extraction), then independently times SAE L29 and dense L22 controllers.
`latency_selected_route.py` uses the **already exported** test probabilities,
validation-tuned thresholds, and existing BM25 timings to calculate total
latency per measured question.

## GPU machine / prerequisites

Use `ml.g6.2xlarge` (8 vCPUs, 32 GiB, 1 NVIDIA L4), matching the earlier L4
run. Bigger single-L4 G6 instance sizes do not increase GPU capacity. Select
an existing Studio space with persistent storage where the data/features
files and cached Hugging Face model/SAE already exist.

Install nothing, and do not run the full `scripts/sagemaker/run.sh`. Use the
existing `.venv/bin/python` and original model/SAE caches.

If the SageMaker repo contains uncommitted work, **do not switch branches or
reset it**. Fetch the new files by `git show` from `origin/final-evaluation`.

## Commands in SageMaker terminal

```bash
cd /home/sagemaker-user/lma-sae-rag-proj
git fetch origin final-evaluation
for f in scripts/sagemaker/latency_selected_measure.py scripts/sagemaker/latency_selected_route.py; do
  git show "origin/final-evaluation:$f" > "$f"
done
export PYTHONPATH=src
.venv/bin/python scripts/sagemaker/latency_selected_measure.py --config configs/sagemaker.yaml --preflight
```

Check that the original model cache exists, and run:

```bash
nvidia-smi
.venv/bin/python scripts/sagemaker/latency_selected_measure.py \
  --config configs/sagemaker.yaml --variant miss_as_retrieve \
  --sae-layer 29 --dense-layer 22 --files nq_test triviaqa_test --n 100 --out-name l29
.venv/bin/python scripts/sagemaker/latency_selected_route.py \
  --config configs/sagemaker.yaml --variant miss_as_retrieve \
  --sae-layer 29 --dense-layer 22 --stage-file stages_l29.jsonl --out routed_l29.csv
```

The measurement script writes `runs/latency/stages_l29.jsonl`,
`runs/latency/stages_l29_summary.csv`, and
`runs/latency/stages_l29_metadata.json`. The routing script writes
`runs/latency/routed_l29.csv`.

**If the stage preflight fails**, do not rerun extraction. Verify the original
SageMaker persistent volume and ask whoever owns it where the features live.
If `runs/latency/bm25.jsonl` is missing, restore it from
`results/sagemaker/latency/bm25.jsonl` if available. Both must use identical
question IDs. If outputs already exist, the GPU script refuses to overwrite;
pass `--overwrite` only if deliberately remeasuring the same environment.

## Export

```bash
mkdir -p results/sagemaker/latency
cp runs/latency/stages_l29.jsonl runs/latency/stages_l29_summary.csv \
  runs/latency/stages_l29_metadata.json runs/latency/routed_l29.csv results/sagemaker/latency/
```

Return these **four exported files** and terminal output. They are enough for
an ADA checkout to import results and regenerate `scripts/finalize_project.py`.

## Validity caveat

`routed_l29.csv` is a **composed estimate from independently measured stages**:
GPU model/probe/B4/answer generation + BM25 lookup on the ADA machine. It is
not an integrated stopwatch measurement of a live serving system. The SAE and
dense controllers are measured separately with layer-specific representations,
but they reuse the recorded generation-time distribution and per-question
retrieval timings. The 200 questions are a convenience sample (first 100 NQ
and first 100 TriviaQA test questions), not a random whole-benchmark sample.
