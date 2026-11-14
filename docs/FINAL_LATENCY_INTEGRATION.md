# Integrating validation-selected L29/L22 controller latency

Source: `l29_latency_outputs.tar.gz` supplied by the SageMaker team on 2026-11-14.

## Locked experiment

- SAE: `gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2`.
- Dense comparator: `gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2`.
- GPU: NVIDIA L4, model `unsloth/gemma-3-4b-it`.
- 200 timed questions: first 100 `nq_test`, first 100 `triviaqa_test`, batch size 1.
- SAE and dense controller stages timed independently. Outputs include closed-book, BM25-conditioned and verbalised-router generation stages.
- Existing BM25 timings from `results/sagemaker/latency/bm25.jsonl` (the original ADA retrieval benchmark) and validation-tuned saved action probabilities were used to calculate *composed*, rather than integrated serving, latencies.

## Outputs

- `results/sagemaker/latency/stages_l29.jsonl`
- `results/sagemaker/latency/stages_l29_summary.csv`
- `results/sagemaker/latency/stages_l29_metadata.json`
- `results/sagemaker/latency/routed_l29.csv`

No additional GPU run is required for this stage of the experiment. Keep the raw SageMaker feature arrays and model cache in their existing workspace.

## Result and interpretation

| Method | Median (ms) | p95 (ms) | Mean (ms) | Retrieval call rate |
|---|---:|---:|---:|---:|
| B1 always retrieve | 231.305 | 492.808 | 273.714 | 100.0% |
| B4 verbalised | 427.036 | 718.247 | 438.283 | 59.0% |
| SAE L29 | 234.440 | 466.645 | 229.070 | 39.0% |
| Dense L22 | 242.647 | 465.789 | 247.462 | 47.0% |

H4 originally hypothesized *lower median latency* than both always-retrieve and generation-time routing. **H4 is not met:** SAE p50 is 3.135 ms slower than always-retrieve. However, SAE mean latency is 16.3% lower, and p95 latency is about 26.2 ms lower than always-retrieve. Report these as secondary descriptive findings, not a revised primary hypothesis.

These results should not be described as a fully integrated end-to-end serving benchmark. The first-100-per-dataset sample is not random and does not establish full-test latency performance.

The finalizer in `scripts/finalize_project.py` verifies both selected layers and all seven measured methods, fixes the NumPy Boolean JSON issue, and includes H4 findings in `results/final/`.

## ADA integration

From `~/saerag-person1/team-repo` on the `final-evaluation` branch, extract the integration patch, then use your existing Python environment to regenerate and test:

```bash
PYTHONPATH=src ~/saerag-person1/.venv/bin/python scripts/finalize_project.py
PYTHONPATH=src ~/saerag-person1/.venv/bin/python -m pytest -q
git diff --check
```

Stage the four `results/sagemaker/latency/stages_l29*` and `routed_l29.csv` files, the finalizer script, generated `results/final/` outputs and this documentation; leave raw `runs/`, `.tar.gz` and Slurm logs untracked.
