#!/usr/bin/env bash
# Side-by-side demo video: Diffusion 1 step | Diffusion 100 steps | Flow 1 step, same PushT seeds.
# scripts/demo.sh <diffusion_ckpt> <flow_ckpt> [episodes] [device]
set -euo pipefail
cd "$(dirname "$0")/.."
DIFF=$1; FLOW=$2; N=${3:-4}; DEV=${4:-mps}; OUT=results/demo
ev() {  # ev <name> <ckpt> <extra args...>
  local name=$1 ckpt=$2; shift 2
  [ -d "$OUT/$name/videos" ] || lerobot-eval --policy.path="$ckpt" --policy.device="$DEV" --env.type=pusht \
    --eval.n_episodes="$N" --eval.batch_size=1 --eval.use_async_envs=false --seed=100000 \
    --output_dir="$OUT/$name" "$@"
}
ev diff1 "$DIFF" --policy.num_inference_steps=1
ev diff100 "$DIFF" --policy.num_inference_steps=100
ev flow1 "$FLOW" --policy.num_inference_steps=1
# Label banners via PIL (Homebrew ffmpeg has no drawtext).
python - "$OUT" <<'EOF'
import sys
from PIL import Image, ImageDraw, ImageFont
for name, text in [("diff1", "Diffusion | 1 step"), ("diff100", "Diffusion | 100 steps"), ("flow1", "Flow (ours) | 1 step")]:
    im = Image.new("RGB", (384, 40), (20, 33, 61))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((12, 8), text, fill=(246, 245, 241), font=font)
    im.save(f"{sys.argv[1]}/label_{name}.png")
EOF
for i in $(seq 0 $((N - 1))); do
  args=(); filt=""; j=0
  for name in diff1 diff100 flow1; do
    args+=(-i "$(ls "$OUT/$name"/videos/*/eval_episode_${i}.mp4)" -loop 1 -i "$OUT/label_$name.png")
    filt+="[$((2 * j)):v]scale=384:384[s$j];[$((2 * j + 1)):v][s$j]vstack=shortest=1[v$j];"
    j=$((j + 1))
  done
  ffmpeg -y -loglevel error "${args[@]}" -filter_complex "${filt}[v0][v1][v2]hstack=3:shortest=0" "$OUT/episode_${i}.mp4"
done
printf "file 'episode_%d.mp4'\n" $(seq 0 $((N - 1))) > "$OUT/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$OUT/list.txt" -c copy "$OUT/demo.mp4"
ffmpeg -y -loglevel error -i "$OUT/demo.mp4" -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$OUT/demo.gif"
echo "$OUT/demo.mp4 $OUT/demo.gif"
