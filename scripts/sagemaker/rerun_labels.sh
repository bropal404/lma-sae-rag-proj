#!/bin/bash
# Re-run everything that depends on the labels after the Gate 1 answer-matcher fix
# (src/saerag/answers.py). Run it on the SageMaker space that already holds the features:
#
#   cd ~/lma-sae-rag-proj && git pull origin aws
#   nohup bash scripts/sagemaker/rerun_labels.sh > logs/rerun.log 2>&1 &
#   tail -f logs/rerun.log
#
# Nothing is generated again and no features are extracted again:
#   1. the stored generations are re-matched with the fixed matcher (old copy in data/generations_prev/)
#   2. the old labels and label-dependent outputs are moved to *_prev/ folders
#   3. the .done markers of the label-dependent steps are removed and run.sh resumes,
#      so it rebuilds labels, Gate 0, baselines, every probe grid, ablation, analysis,
#      scores and routed latency. GPU steps (generate, features, wide262k, b4, latency)
#      and bm25_latency stay as they are.
# Settings are the same environment variables as run.sh (PRIMARY, LAYER, SEEDS, BRANCH, ...).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs runs/.done
PY=.venv/bin/python

for need in data/generations runs/.done/features; do
  [ -e "$need" ] || { echo "missing $need: this space has no finished first run; use run.sh instead"; exit 1; }
done

if [ -f runs/.done/rescored ]; then
  echo "== generations already rescored, skipping"
else
  echo "== $(date '+%F %T') rescoring generations with the fixed matcher"
  $PY scripts/rescore_generations.py
  touch runs/.done/rescored
fi

# Keep the old label-dependent outputs for comparison, once.
stamp=prev_matcher_v1
for d in data/labels runs/routers runs/probes runs/scores runs/reports runs/diagnostics; do
  if [ -d "$d" ] && [ ! -e "${d}_${stamp}" ]; then
    mv "$d" "${d}_${stamp}"
    echo "moved $d -> ${d}_${stamp}"
  fi
done
# B4's probabilities come from the model, not the labels: put them back so B4 is not re-run on the GPU.
if [ -d "runs/routers_${stamp}/b4_verbalised" ] && [ ! -d runs/routers/b4_verbalised ]; then
  mkdir -p runs/routers
  cp -r "runs/routers_${stamp}/b4_verbalised" runs/routers/
fi

# The handcheck step is kept: re-running it would overwrite the hand-checked sheet.
for s in labels gate0 baselines anomaly grid_main dense_extra grid_other twoway score routed_latency \
         analyze cgrid ablate_l1 ablate_bin ablate_mean probes_262k score2 routed_latency2; do
  rm -f "runs/.done/$s"
done
echo "== markers left: $(ls runs/.done | tr '\n' ' ')"

exec bash scripts/sagemaker/run.sh
