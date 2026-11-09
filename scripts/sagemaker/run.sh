#!/bin/bash
# Full remaining run on one SageMaker GPU space (tested target: ml.g6.2xlarge, one L4).
#
#   git clone -b aws https://<token>@github.com/bropal404/lma-sae-rag-proj.git ~/lma-sae-rag-proj
#   cd ~/lma-sae-rag-proj && git remote set-url origin https://github.com/bropal404/lma-sae-rag-proj.git
#   echo "GH_TOKEN=<token>" > .env            # or export GH_TOKEN; .env is gitignored
#   nohup bash scripts/sagemaker/run.sh > logs/run.log 2>&1 &
#   tail -f logs/run.log
#
# The token needs Contents read and write on the repo: small result files are committed to
# results/sagemaker/ and pushed to $BRANCH after the GPU stage, at the end, and if a step fails.
# Each finished step leaves runs/.done/<step>, so running the script again resumes.
# Settings (environment variables): PRIMARY label rule, LAYER probe layer, SEEDS, BRANCH,
# COMMIT_RESULTS=0 to skip git, GIT_NAME / GIT_EMAIL for the commits.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs runs/.done

PRIMARY=${PRIMARY:-miss_as_retrieve}
if [ "$PRIMARY" = default ]; then SECONDARY=miss_as_retrieve; else SECONDARY=default; fi
LAYER=${LAYER:-17}
SEEDS=${SEEDS:-"0 1 2"}
BRANCH=${BRANCH:-aws}
COMMIT_RESULTS=${COMMIT_RESULTS:-1}
export HF_HOME=${HF_HOME:-$HOME/hf} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
PY=.venv/bin/python
C="--config configs/sagemaker.yaml"
ALL="pilot nq_train nq_val nq_test triviaqa_train triviaqa_val triviaqa_test popqa_test hotpotqa_test selfaware_test"
TRAIN="nq_train triviaqa_train"
EVAL="nq_val triviaqa_val nq_test triviaqa_test popqa_test hotpotqa_test selfaware_test"
TEST="nq_test triviaqa_test popqa_test hotpotqa_test selfaware_test"
LAYERS="9 17 22 29"
WIDE_LAYERS="17 22"
TAG=gemma3-4b-it

# ---- token: GH_TOKEN from the environment, else .env (GH_TOKEN=... or the bare token) ----
if [ -z "${GH_TOKEN:-}" ] && [ -f .env ]; then
  GH_TOKEN=$(grep -m1 -E '^(GH_TOKEN=)?(github_pat_|ghp_)' .env | sed 's/^GH_TOKEN=//' || true)
fi
export GH_TOKEN=${GH_TOKEN:-}
if [ "$COMMIT_RESULTS" = 1 ] && [ -z "$GH_TOKEN" ]; then
  echo "no GH_TOKEN: results will be committed locally but not pushed"
fi
git config user.name "${GIT_NAME:-Gopal Kataria}"
git config user.email "${GIT_EMAIL:-gopal.kataria@research.iiit.ac.in}"

# ---- machine ----
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
echo "free disk in home: ${free_gb} GB"
if [ "$free_gb" -lt 15 ]; then
  echo "Need at least 15 GB free: stop the space, raise its storage, start again."; exit 1
fi
need_gb() {   # need_gb N: fail the current step unless N GB are free
  local free; free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$1" ] || { echo "only ${free} GB free, this step needs $1 GB: raise the space's storage"; return 1; }
}

# ---- Python: torch to match the driver (the lockfile's torch needs CUDA 13), then the project ----
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x $PY ] || uv venv --python 3.12 .venv
cuda_ok() { $PY -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()" 2>/dev/null; }
if ! cuda_ok; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "$driver" -ge 580 ]; then flavours="cu130 cu128"; else flavours="cu128 cu126"; fi
  for fl in $flavours; do
    uv pip install --python $PY --reinstall-package torch torch --index-url https://download.pytorch.org/whl/$fl \
      && cuda_ok && { echo "torch build: $fl"; break; }
  done
  cuda_ok || { echo "No torch build sees the GPU"; exit 1; }
fi
uv pip install --python $PY -e ".[features]" "transformers==5.19.0" "accelerate==1.15.0"
$PY -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"

# ---- data and weights (ungated mirror of google/gemma-3-4b-it, same weight hashes) ----
if [ ! -f data/questions/pilot.jsonl ]; then
  mkdir -p data
  tar xzf artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz -C data questions retrieval
fi
$PY -m saerag.prefetch $C --widths 16k


step() {
  local name=$1; shift
  if [ -f "runs/.done/$name" ]; then echo "== skip $name (done)"; return; fi
  echo "== $(date '+%F %T') start $name"
  local t0=$SECONDS
  "$@" || { echo "== $name FAILED"; return 1; }
  touch "runs/.done/$name"
  echo "== $name done in $(( (SECONDS - t0) / 60 )) min"
}
gate0() { $PY -m saerag.generation.report $C --files pilot || echo "Gate 0: a class is under 15% under the default rule (see runs/reports/)"; }
publish() {   # copy the small result files to results/sagemaker/, commit, push to $BRANCH
  [ "${COMMIT_RESULTS:-1}" = 1 ] || return 0
  local out=results/sagemaker
  mkdir -p $out
  ( cd runs && find reports scores latency diagnostics handcheck -type f 2>/dev/null | xargs -r cp --parents -t ../$out ) || true
  ( cd runs && find probes -type f \( -name summary.csv -o -name 'top_features_*.json' -o -path '*analysis__*' \) 2>/dev/null | xargs -r cp --parents -t ../$out ) || true
  ( cd data && find features generations -type f \( -name 'stats_*.json' -o -name '*.stats.json' \) 2>/dev/null | xargs -r cp --parents -t ../$out ) || true
  if [ -f logs/run_all.log ]; then cp logs/run_all.log $out/; fi
  git add $out
  if git diff --cached --quiet; then echo "publish: nothing new"; return 0; fi
  git commit -q -m "SageMaker results: $1" -m "Run on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), label rule $PRIMARY, probe layer $LAYER."       -m "Co-Authored-By: Amazon SageMaker <sagemaker@noreply.invalid>"
  echo "publish: committed $1"
  if [ -n "${GH_TOKEN:-}" ]; then
    export GIT_ASKPASS=$PWD/scripts/sagemaker/askpass.sh GIT_TERMINAL_PROMPT=0
    { git pull -q --rebase origin "$BRANCH" && git push -q origin HEAD:"$BRANCH" && echo "publish: pushed to $BRANCH"; }       || echo "publish: push failed; the commit is still local"
  fi
}
score_all() {
  local routers="b1_always_retrieve b2_never_retrieve b3_oracle b4_verbalised b8_tfidf_lr"
  for L in $LAYERS; do for k in resid_last sae16k_last; do
    routers="$routers ${TAG}__${PRIMARY}__${k}@L${L}__l2 ${TAG}__${PRIMARY}__${k}@L${L}__l2__2way"
  done; done
  for L in $WIDE_LAYERS; do
    [ -d "runs/routers/${TAG}__${PRIMARY}__sae262k_last@L${L}__l2__wide" ] && routers="$routers ${TAG}__${PRIMARY}__sae262k_last@L${L}__l2__wide"
  done
  for f in $TEST; do
    local val=nq_val; [ "$f" = triviaqa_test ] && val=triviaqa_val
    $PY -m saerag.eval.score $C --routers $routers --files $f --variant $PRIMARY --val-file $val
  done
}

BRANCH=${BRANCH:-aws}
trap 'publish "partial, a step failed" || true' ERR
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "main label rule: $PRIMARY   probe layer: $LAYER   seeds: $SEEDS"

# ---- GPU ----
step generate   $PY -m saerag.generation.run $C --condition closed bm25 gold --files $ALL --backend hf
step labels     $PY -m saerag.generation.make_labels $C --files $ALL
step gate0      gate0
step features   $PY -m saerag.features.extract $C --files $ALL --widths 16k
step b4         $PY -m saerag.baselines.run $C --routers b4 --files $EVAL --backend hf --variant $PRIMARY
step latency    $PY scripts/sagemaker/latency_measure.py $C --layer $LAYER --variant $PRIMARY
publish "GPU stage (generations, labels, features, B4, latency)" || true

# ---- CPU ----
step handcheck  $PY -m saerag.eval.handcheck $C sample --files nq_test triviaqa_test popqa_test hotpotqa_test --n 200 --variant $PRIMARY
step baselines  $PY -m saerag.baselines.run $C --routers b1 b2 b3 b8 --files $EVAL --variant $PRIMARY
step anomaly    $PY scripts/sagemaker/anomaly.py $C --layer $LAYER --labels closed $PRIMARY --seeds $SEEDS
step grid_main  $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY \
                    --layers $LAYERS --kinds resid_last sae16k_last --budgets 50 100 200 500 1000 2000 5000 --seeds $SEEDS
step dense_extra $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY \
                    --layers 4 13 --kinds resid_last --name extra_layers
step grid_other $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $SECONDARY \
                    --layers $LAYERS --kinds resid_last sae16k_last
step twoway     $PY scripts/sagemaker/twoway.py $C --variant $PRIMARY --layers $LAYERS --files $EVAL
step score      score_all
step routed_latency $PY scripts/sagemaker/latency_route.py $C --variant $PRIMARY --layer $LAYER

# ---- second session: wider SAEs, BM25 latency, ablations, feature analysis ----
wide262k() { need_gb 15 && $PY -m saerag.features.extract $C --files $ALL --sae-layers $WIDE_LAYERS --widths 262k --dense-layers; }
bm25_env() {   # separate CPU environment, so Pyserini's dependencies cannot touch the GPU torch
  need_gb 25 || return 1
  local jdk=$HOME/jdk21
  if [ ! -x $jdk/bin/java ]; then
    mkdir -p $jdk
    curl -fsSL "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse" | tar xz -C $jdk --strip-components=1
  fi
  export JAVA_HOME=$jdk PATH=$jdk/bin:$PATH
  [ -x .venv-bm25/bin/python ] || uv venv --python 3.12 .venv-bm25
  uv pip install --python .venv-bm25/bin/python torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-bm25/bin/python -e . "pyserini==2.4.0"
  .venv-bm25/bin/python scripts/sagemaker/bm25_latency.py $C
}
analyze() {
  for L in $WIDE_LAYERS; do
    $PY -m saerag.probes.analyze $C --train $TRAIN --eval nq_test triviaqa_test popqa_test hotpotqa_test \
        --labels $PRIMARY --layer $L --kind sae16k_last || return 1
  done
}

step wide262k   wide262k || echo "262k extraction skipped"
publish "262k SAE features" || true
step analyze    analyze || echo "analyze skipped"
step cgrid      $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY --layers $WIDE_LAYERS \
                    --kinds resid_last --c-grid 0.00001 0.00003 0.0001 0.0003 0.001 0.01 --name cgrid || echo "cgrid skipped"
step ablate_l1  $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY --penalty l1 \
                    --layers $WIDE_LAYERS --kinds sae16k_last --budgets 1000 --seeds $SEEDS --name l1 || echo "ablate_l1 skipped"
step ablate_bin $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY --binarize \
                    --layers $WIDE_LAYERS --kinds sae16k_last --budgets 1000 --seeds $SEEDS --name binarized || echo "ablate_bin skipped"
step ablate_mean $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY \
                    --layers $WIDE_LAYERS --kinds resid_mean sae16k_mean --budgets 1000 --seeds $SEEDS --name mean_pool || echo "ablate_mean skipped"
if [ -f runs/.done/wide262k ]; then
  step probes_262k $PY -m saerag.probes.train $C --train $TRAIN --eval $EVAL --labels $PRIMARY --layers $WIDE_LAYERS \
                      --kinds sae262k_last --budgets 50 100 200 500 1000 2000 5000 --seeds $SEEDS --name wide || true
fi
step score2     score_all || echo "score2 skipped"
step bm25_latency bm25_env || echo "BM25 latency skipped"
step routed_latency2 $PY scripts/sagemaker/latency_route.py $C --variant $PRIMARY --layer $LAYER || echo "routed_latency2 skipped"

tar czf results_bundle.tar.gz runs data/labels data/generations data/features/$TAG/stats_*.json logs
publish "full run" || true
echo "== all done: results_bundle.tar.gz"
