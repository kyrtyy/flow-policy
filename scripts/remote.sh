#!/usr/bin/env bash
# Orchestrate the 4090 server from the Mac.
#   scripts/remote.sh sync                 push code (no venv/outputs)
#   scripts/remote.sh setup                sync + create venv on server
#   scripts/remote.sh run <name> <cmd...>  run cmd detached, log to logs/<name>.log ("EXIT <code>" at the end)
#   scripts/remote.sh logs <name>          tail a run's log
#   scripts/remote.sh kill <name>          stop a run and its children
#   scripts/remote.sh gpu                  nvidia-smi (is the server free?)
#   scripts/remote.sh pull                 fetch results/ back
set -euo pipefail
HOST=${HOST:-yashdewangan@10.81.30.91}  # 3050; the 4090 is yashdewangan@10.24.32.171
DIR=${DIR:-flowpolicy}
cd "$(dirname "$0")/.."

sync() { rsync -az --delete --exclude .venv --exclude outputs --exclude ckpt --exclude full --exclude results --exclude logs \
  --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude requirements.lock \
  ./ "$HOST:$DIR/"; }

case ${1:-} in
  sync) sync ;;
  # The lock is written on the server (CUDA wheels); copy it back so the pinned versions live with the code.
  setup) sync; ssh "$HOST" "cd $DIR && bash scripts/setup_server.sh" && rsync -az "$HOST:$DIR/requirements.lock" . ;;
  run)
    # nohup+setsid instead of tmux: survives logout and needs nothing installed on the server.
    name=$2; shift 2; sync
    ssh "$HOST" "cd $DIR && mkdir -p logs && { nohup setsid bash -c \
      'source .venv/bin/activate && ($*); echo EXIT \$?' > logs/$name.log 2>&1 < /dev/null & echo \$! > logs/$name.pid; }"
    echo "started '$name' — scripts/remote.sh logs $name" ;;
  logs) ssh "$HOST" "tail -n ${3:-40} $DIR/logs/$2.log" ;;
  kill) ssh "$HOST" "kill -- -\$(cat $DIR/logs/$2.pid)" ;;
  gpu) ssh "$HOST" nvidia-smi ;;
  pull) rsync -az "$HOST:$DIR/results/" results/ ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
