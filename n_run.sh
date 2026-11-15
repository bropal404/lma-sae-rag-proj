#!/usr/bin/bash
set -euo pipefail   # stop on first error

# Load .env into the environment (git can't read it otherwise)
set -a; source .env; set +a

# uv run scripts/sagemaker/latency_selected_measure.py --config configs/sagemaker.yaml --preflight

# Add --overwrite only if you really want to redo the stage measurements
# uv run scripts/sagemaker/latency_selected_measure.py --config configs/sagemaker.yaml \
#   --variant miss_as_retrieve --sae-layer 29 --dense-layer 22 \
#   --files nq_test triviaqa_test --n 100 --out-name l29

# uv run scripts/sagemaker/latency_selected_route.py --config configs/sagemaker.yaml \
#   --variant miss_as_retrieve --sae-layer 29 --dense-layer 22 \
#   --stage-file stages_l29.jsonl --out routed_l29.csv

tar -czf l29_latency_outputs.tar.gz -C runs/latency \
  stages_l29.jsonl stages_l29_summary.csv stages_l29_metadata.json routed_l29.csv

# Refuse to continue if .env would be committed
git check-ignore -q .env || { echo ".env is not gitignored!"; exit 1; }

git add .
git commit -m "L29 latency outputs" || echo "nothing to commit"

# Pass the token through a one-off credential helper (not stored, not in the URL)
git -c credential.helper= \
    -c credential.helper='!f() { echo username=bropal404; echo "password=${GITHUB_TOKEN}"; }; f' \
    push origin HEAD