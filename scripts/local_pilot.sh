#!/bin/bash
# Small pilot on this machine: Gemma 3 1B-IT on CPU, needs no login and no GPU.
# Checks the plumbing and shows the shape of the results. 1B knows less than 4B, so these
# numbers are not for the report and do not answer Gate 0.
#
#   bash scripts/local_pilot.sh             # 500 questions, about 40 minutes
#   LIMIT=100 bash scripts/local_pilot.sh   # about 8 minutes, NQ only (the file starts with NQ)
#
# Needs about 5 GB of free RAM, so close Chrome first. Results stay out of data/ in
# data/local_1b/, because generation paths do not carry the model name.

set -euo pipefail
[ -f configs/cpu_1b.yaml ] || { echo "run this from the repo root"; exit 1; }
LIMIT=${LIMIT:-500}
export SAERAG_DATA=$PWD/data/local_1b
export LOKY_MAX_CPU_COUNT=4         # the probe search otherwise starts one worker per core
mkdir -p "$SAERAG_DATA"
ln -sfn "$PWD/data/questions" "$SAERAG_DATA/questions"
ln -sfn "$PWD/data/retrieval" "$SAERAG_DATA/retrieval"
RUN="uv run --no-sync python -m"
C="--config configs/cpu_1b.yaml"

$RUN saerag.prefetch $C --widths 16k
$RUN saerag.generation.run $C --condition closed bm25 gold --files pilot --backend hf \
    --limit "$LIMIT" --overwrite
$RUN saerag.generation.make_labels $C --files pilot
$RUN saerag.generation.report $C --files pilot || echo "gate failed (expected often on 1B)"
$RUN saerag.features.extract $C --files pilot --limit "$LIMIT" --sae-layers 13 17 --dense-layers 13 17
$RUN saerag.probes.train $C --train pilot --labels closed --layers 13 17 --budgets 50
echo "report: runs/reports/gemma3-1b-it/pilot.md   probes: runs/probes/gemma3-1b-it/closed/"
