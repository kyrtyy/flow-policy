#!/usr/bin/env bash
# Full version: published-size config, 200k steps, 3 seeds x 4 policies, then the eval sweep.
# Runs $QUEUES training queues spread over $NGPU GPUs (default 4 queues on 2 GPUs = 2 jobs per GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-200000}; NGPU=${NGPU:-2}; QUEUES=${QUEUES:-4}
# Heavy jobs first so they spread across queues.
JOBS=()
for p in diffusion flow act bcmlp; do for s in 1000 1001 1002; do JOBS+=("$p $s"); done; done

worker() {  # worker <queue index>
  local q=$1 i
  for (( i=q; i<${#JOBS[@]}; i+=QUEUES )); do
    read -r p s <<<"${JOBS[$i]}"
    [ -f "outputs/train/${p}_s${s}_${STEPS}/checkpoints/last/pretrained_model/model.safetensors" ] && continue
    CUDA_VISIBLE_DEVICES=$(( q % NGPU )) WORKERS=6 scripts/train.sh "$p" "$s" "$STEPS" > "logs/train_${p}_s${s}.log" 2>&1 \
      && echo "DONE $p s$s" || echo "FAILED $p s$s (logs/train_${p}_s${s}.log)"
  done
}
for (( q=0; q<QUEUES; q++ )); do worker "$q" & done
wait
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sweep.py outputs/train/*_"${STEPS}" --out results/full
python scripts/aggregate.py results/full
