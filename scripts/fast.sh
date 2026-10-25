#!/usr/bin/env bash
# Fast version on one GPU.
#   scripts/fast.sh bench   500-step timing run per policy -> real it/s to size the full run
#   scripts/fast.sh all     train 4 policies (1 seed, $STEPS steps) then run the eval sweep
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-40000}
POLICIES=(bcmlp act diffusion flow)

case ${1:-all} in
  bench)
    for p in "${POLICIES[@]}"; do
      s=$SECONDS; scripts/train.sh "$p" 0 500 --output_dir="outputs/bench/$p" --save_checkpoint=false
      echo "BENCH $p: $(( SECONDS - s ))s for 500 steps (incl. startup)"
    done ;;  # summary: scripts/remote.sh logs <name> 400 | grep BENCH
  all)
    # ponytail: sequential so we hold one GPU slot on the shared server; run in parallel if it's ours.
    for p in "${POLICIES[@]}"; do scripts/train.sh "$p" 1000 "$STEPS"; done
    python scripts/eval_sweep.py $(printf "outputs/train/%s_s1000_${STEPS} " "${POLICIES[@]}") --out results/fast ;;
esac
