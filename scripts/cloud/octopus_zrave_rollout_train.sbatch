#!/usr/bin/env bash
#SBATCH --job-name=zrave-rollout-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yfhuang/zrave-rollout-train-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50_rollout.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
run_root=$root/runs/zrave_transformer_rollout_v3
container_run=$container_root/runs/zrave_transformer_rollout_v3
warm_start=$root/runs/zrave_transformer_mean50_v2/checkpoints/best.pt
container_warm_start=$container_root/runs/zrave_transformer_mean50_v2/checkpoints/best.pt
selection=${SWEEP_SELECTION:-$root/sweeps/rollout-v3/latest-selection.json}

test -f "$selection"
test -f "$warm_start"
batch=$(python3 - "$selection" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1]))["batch_per_gpu"]))
PY
)
case "$batch" in
  512|1024|1536|2048) ;;
  *)
    echo "invalid selected batch: $batch" >&2
    exit 2
    ;;
esac

mkdir -p "$run_root/checkpoints" "$run_root/tensorboard"
start_args=()
resume=${ZRAVE_RESUME:-}
if [[ -n "$resume" && "$resume" != "$(basename "$resume")" ]]; then
  echo "ZRAVE_RESUME must be a checkpoint filename, got: $resume" >&2
  exit 3
fi
if [[ -z "$resume" ]]; then
  resume=$(find "$run_root/checkpoints" -maxdepth 1 -type f \
    -name 'update-*.pt' -printf '%f\n' | sort | tail -n 1 || true)
fi
if [[ -n "$resume" ]]; then
  test -f "$run_root/checkpoints/$resume"
  start_args=(--resume "$container_run/checkpoints/$resume")
else
  start_args=(--warm-start "$container_warm_start")
fi

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=32g \
    -v "$repo:$container_repo:ro" \
    -v "$root:$container_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -e NCCL_DEBUG=WARN \
    -w "$container_repo" \
    "$image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_train \
      --config "$config" \
      --batch-per-gpu "$batch" \
      --max-updates 3000 \
      "${start_args[@]}"
