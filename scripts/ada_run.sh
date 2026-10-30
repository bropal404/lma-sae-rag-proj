#!/bin/bash
# One command for ADA. Run on the login node from the repo root; it sets everything up and
# submits the jobs, so you can log out afterwards.
#
#   bash scripts/ada_run.sh pilot    # Gate 0: 500 questions on GPU, about an hour
#   bash scripts/ada_run.sh full     # every split, then the CPU probe grid
#
# Optional settings, as environment variables:
#   DATA_ROOT=/home2/$USER/saerag    must be a path the compute nodes can see (default $HOME/saerag)
#   ACCOUNT=<slurm account>          PARTITION=<partition>
#   GPUS=2                           2 on 2080 Ti / T4 (no bf16), 1 on A100 / 3090
#   WIDE=1                           also fetch and extract the 262k SAEs (about 21 GB more)
#
# Needs once, by hand: accept the licence at https://huggingface.co/google/gemma-3-4b-it,
# then `hf auth login`. Space: about 15 GB, or 36 GB with WIDE=1.

set -euo pipefail
MODE=${1:?usage: bash scripts/ada_run.sh pilot|full}
[ -f scripts/ada_tonight.sbatch ] || { echo "run this from the repo root"; exit 1; }

export DATA_ROOT=${DATA_ROOT:-$HOME/saerag}
export SAERAG_DATA=$DATA_ROOT/data
export HF_HOME=$DATA_ROOT/hf
export WIDE=${WIDE:-0}
GPUS=${GPUS:-2}
SLURM_ARGS=()
[ -n "${ACCOUNT:-}" ] && SLURM_ARGS+=(-A "$ACCOUNT")
[ -n "${PARTITION:-}" ] && SLURM_ARGS+=(-p "$PARTITION")

# ---- setup: code environment, data, model. Everything here is safe to repeat. ----
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
uv sync --extra features
mkdir -p "$SAERAG_DATA" "$HF_HOME" logs
df -h "$DATA_ROOT" | tail -1
if [ ! -f "$SAERAG_DATA/questions/pilot.jsonl" ]; then
  tar xzf artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz -C "$SAERAG_DATA" questions retrieval
fi
# Anything made from the old question set has different qids (D13).
first_qid=$(head -1 "$SAERAG_DATA/questions/pilot.jsonl" | cut -d'"' -f4)
old=$(ls "$SAERAG_DATA"/generations/closed/pilot.jsonl "$SAERAG_DATA"/features/*/qids_pilot.json 2>/dev/null || true)
if [ -n "$old" ] && ! grep -q "$first_qid" $old; then
  echo "$SAERAG_DATA has results from another question set; delete generations/ labels/ features/"; exit 1
fi
uv run --no-sync hf auth whoami >/dev/null || { echo "run: uv run --no-sync hf auth login"; exit 1; }
WIDTHS=16k; [ "$WIDE" = 1 ] && WIDTHS="16k 262k"
uv run --no-sync python -m saerag.prefetch --widths $WIDTHS

# ---- submit ----
if [ "$MODE" = pilot ]; then
  sbatch "${SLURM_ARGS[@]}" -J saerag-pilot -c 16 --gres=gpu:"$GPUS" --mem-per-cpu=3G --time=4:00:00 \
      -o "logs/%x-%j.out" <<'JOB'
#!/bin/bash
set -euo pipefail
export HF_HUB_OFFLINE=1
RUN="uv run --no-sync python -m"
nvidia-smi
$RUN saerag.generation.run --condition closed bm25 gold --files pilot --backend hf
$RUN saerag.generation.make_labels --files pilot
$RUN saerag.generation.report --files pilot || echo "GATE 0 FAILED"
$RUN saerag.features.extract --files pilot --widths 16k
$RUN saerag.probes.train --train pilot --labels closed --layers 9 17 22 29 \
    --kinds resid_last sae16k_last --budgets 50 100
$RUN saerag.probes.train --train pilot --labels default --layers 9 17 22 29 \
    --kinds resid_last sae16k_last --budgets 100
JOB
elif [ "$MODE" = full ]; then
  first=$(sbatch --parsable "${SLURM_ARGS[@]}" --gres=gpu:"$GPUS" scripts/ada_tonight.sbatch)
  second=$(sbatch --parsable "${SLURM_ARGS[@]}" --dependency=afterok:"$first" scripts/ada_grid.sbatch)
  echo "submitted $first (GPU), then $second (CPU grid)"
else
  echo "usage: bash scripts/ada_run.sh pilot|full"; exit 1
fi
echo "watch: squeue -u \$USER, then logs/. Reports: runs/reports/. Data: $SAERAG_DATA"
