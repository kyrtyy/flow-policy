#!/usr/bin/env bash
# Train one policy on PushT.  Usage: scripts/train.sh <diffusion|flow|bcmlp|act> [seed] [steps] [extra lerobot-train args...]
# NAME=<run name> overrides the output dir prefix (e.g. NAME=reflow with --policy.teacher_path=...).
# Fast version default: 40k steps. Full version: 200k (the LeRobot/paper PushT setting).
set -euo pipefail
POLICY=$1; SEED=${2:-1000}; STEPS=${3:-40000}; shift $(( $# < 3 ? $# : 3 ))

# Diffusion Policy's published PushT config (lerobot/diffusion_pusht). Flow and BC-MLP share it so the
# only difference between them is the training objective / head.
DP_ARGS=(--policy.horizon=16 --policy.n_action_steps=8 --policy.n_obs_steps=2
  --policy.crop_shape="[84,84]" --policy.use_group_norm=true --policy.pretrained_backbone_weights=null
  --policy.use_separate_rgb_encoder_per_camera=false)
# ACT: chunk of 16, execute 8, matching the others' execution horizon.
ACT_ARGS=(--policy.chunk_size=16 --policy.n_action_steps=8)

[[ $POLICY == act ]] && ARGS=("${ACT_ARGS[@]}") || ARGS=("${DP_ARGS[@]}")
# SMALL=1: 6 GB GPUs (RTX 3050). 4x smaller U-Net (76M vs 263M params) for diffusion/flow/bcmlp, AMP for all.
# Not the published DP size, so results are pipeline/trend checks, not a reproduction.
if [[ ${SMALL:-0} == 1 ]]; then
  [[ $POLICY == act ]] || ARGS+=(--policy.down_dims="[256,512,1024]")
  ARGS+=(--policy.use_amp=true)
fi

# torchcodec needs system ffmpeg libs; without them it fails and falls back to PyAV in every worker.
command -v ffmpeg >/dev/null || ARGS+=(--dataset.video_backend=pyav)

lerobot-train \
  --policy.type="$POLICY" "${ARGS[@]}" \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --seed="$SEED" --steps="$STEPS" --batch_size="${BATCH:-64}" --num_workers="${WORKERS:-8}" \
  --env_eval_freq=0 --save_freq="$STEPS" --log_freq=200 \
  --wandb.enable="${WANDB:-false}" --wandb.project=flowpolicy \
  --output_dir="outputs/train/${NAME:-$POLICY}_s${SEED}_${STEPS}" \
  "$@"
