#!/bin/bash
# Tonight's GPU run on the IIIT cluster: Person 2 closed-book generations (+ RAG/labels if
# Person 1's retrieval files exist) and Person 3 features + probes on proxy labels.
#
# Before submitting, on the login node (needs internet):
#   uv sync --extra features            # add --extra gpu to use vLLM (BACKEND=vllm)
#   hf auth login                       # after accepting the Gemma 3 licence on Hugging Face
#   export SAERAG_DATA=/scratch/$USER/saerag/data HF_HOME=/scratch/$USER/hf
#   mkdir -p "$SAERAG_DATA" logs
#   tar xzf artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz -C "$SAERAG_DATA" questions retrieval
#   uv run python -m saerag.prefetch --widths 16k 262k
# Then: sbatch scripts/ada_tonight.sbatch     (WIDE=1 sbatch ... adds the 262k SAE passes)
# Finished generation files are skipped, so resubmitting after a timeout resumes the run.
#
# GPUs: 2080 Ti / T4 have no bf16, so the model runs in fp32 (~17 GB) split across GPUs:
# ask for 2 (3 if you also want 262k SAEs). On an A100/3090 node, 1 GPU and BACKEND=vllm.

#SBATCH -J saerag-tonight
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=10:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export SAERAG_DATA=${SAERAG_DATA:-/scratch/$USER/saerag/data}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_OFFLINE=1            # everything was prefetched on the login node
BACKEND=${BACKEND:-hf}             # hf works everywhere; vllm needs `uv sync --extra gpu`
RUN="uv run --no-sync python -m"
mkdir -p logs
nvidia-smi

FILES="nq_train nq_val nq_test triviaqa_train triviaqa_val triviaqa_test popqa_test hotpotqa_test selfaware_test"
SAE_LAYERS="9 17 22 29"

# ---- Person 2: all three conditions, labels, and the pilot gate report ----
$RUN saerag.generation.run --condition closed bm25 gold --files pilot $FILES --backend $BACKEND
$RUN saerag.generation.make_labels --files pilot $FILES
$RUN saerag.generation.report --files pilot || echo "GATE 0 FAILED: read the report before the grid"

# ---- Person 3: activations + 16k SAE codes (one forward pass per question) ----
$RUN saerag.features.extract --files pilot $FILES --widths 16k
if [ "${WIDE:-0}" = 1 ]; then
  for L in $SAE_LAYERS; do
    $RUN saerag.features.extract --files pilot $FILES --sae-layers $L --widths 262k --dense-layers
  done
fi

# ---- Person 3: probes on the proxy label (closed-book correct = ANSWER) ----
$RUN saerag.probes.train --train pilot --labels closed --layers $SAE_LAYERS \
    --kinds resid_last resid_mean sae16k_last sae16k_mean --budgets 50 100
$RUN saerag.probes.train --train nq_train triviaqa_train \
    --eval nq_test triviaqa_test popqa_test hotpotqa_test --labels closed --layers $SAE_LAYERS \
    --kinds resid_last sae16k_last --budgets 50 100 500 1000

# ---- Person 3: first real 3-way probes (the full grid is scripts/ada_grid.sbatch) ----
$RUN saerag.probes.train --train nq_train triviaqa_train \
    --eval nq_val triviaqa_val nq_test triviaqa_test popqa_test hotpotqa_test selfaware_test \
    --labels default --layers $SAE_LAYERS --kinds resid_last sae16k_last --budgets 50 100 500 1000

echo "done"
