#!/bin/bash
# Person 3 probe grid and feature analysis. CPU only; run after ada_tonight.sbatch.
# Layers x widths x label budgets x seeds, SAE and dense (B7), plus pooling and L1 ablations.
# Set WIDE=1 if the 262k features were extracted.

#SBATCH -J saerag-grid
#SBATCH -A <your_account>
#SBATCH -c 32
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export SAERAG_DATA=${SAERAG_DATA:-/scratch/$USER/saerag/data}
RUN="uv run --no-sync python -m"

TRAIN="nq_train triviaqa_train"
EVAL="nq_val triviaqa_val nq_test triviaqa_test popqa_test hotpotqa_test selfaware_test"
LAYERS="9 17 22 29"
BUDGETS="50 100 200 500 1000 2000 5000"
SEEDS="0 1 2 3 4"
KINDS="resid_last sae16k_last"
if [ "${WIDE:-0}" = 1 ]; then KINDS="$KINDS sae262k_last"; fi

# Main grid, both label variants. Dense at the two extra layers is the B7 layer sweep.
for labels in default miss_as_retrieve; do
  $RUN saerag.probes.train --train $TRAIN --eval $EVAL --labels $labels \
      --layers $LAYERS --kinds $KINDS --budgets $BUDGETS --seeds $SEEDS
done
$RUN saerag.probes.train --train $TRAIN --eval $EVAL --labels default \
    --layers 4 13 --kinds resid_last --budgets $BUDGETS --seeds $SEEDS --name extra_layers

# Ablations: L1 penalty, binarised codes, mean pooling (D5).
$RUN saerag.probes.train --train $TRAIN --eval $EVAL --labels default --penalty l1 \
    --layers $LAYERS --kinds sae16k_last --budgets 1000 --seeds $SEEDS
$RUN saerag.probes.train --train $TRAIN --eval $EVAL --labels default --binarize \
    --layers $LAYERS --kinds sae16k_last --budgets 1000 --seeds $SEEDS
$RUN saerag.probes.train --train $TRAIN --eval $EVAL --labels default \
    --layers $LAYERS --kinds resid_mean sae16k_mean --budgets 1000 --seeds $SEEDS --name mean_pool

# Feature analysis: top features, hand-check sheet, drop-top-k.
for L in $LAYERS; do
  $RUN saerag.probes.analyze --train $TRAIN --eval nq_test triviaqa_test --labels default \
      --layer $L --kind sae16k_last
done
echo "done"
