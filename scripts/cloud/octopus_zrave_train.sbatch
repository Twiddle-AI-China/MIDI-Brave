#!/usr/bin/env bash
#SBATCH --job-name=zrave-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yfhuang/zrave-train-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_balanced50.yaml
image=midibrave:v0.10.0-clap-recon
run_root=/data/midibrave-zrave/runs/zrave_transformer_balanced50
selection=${SWEEP_SELECTION:-/data/midibrave-zrave/sweeps/latest-selection.json}

test -f "$selection"
batch=$(python3 - "$selection" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1]))["batch_per_gpu"]))
PY
)
if [[ "$batch" != 128 && "$batch" != 256 && "$batch" != 384 && "$batch" != 512 ]]; then
  echo "invalid selected batch: $batch" >&2
  exit 2
fi

mkdir -p "$run_root/checkpoints" "$run_root/tensorboard"
resume_args=()
resume=$(find "$run_root/checkpoints" -maxdepth 1 -type f \
  -name 'update-*.pt' -printf '%f\n' | sort | tail -n 1 || true)
if [[ -n "$resume" ]]; then
  resume_args=(--resume "$run_root/checkpoints/$resume")
fi

docker run --rm --network host \
  -v /data:/data \
  "$image" \
  tensorboard \
    --logdir "$run_root/tensorboard" \
    --host 0.0.0.0 \
    --port 6006 \
    --reload_interval 15 &
tensorboard_pid=$!
cleanup() {
  kill "$tensorboard_pid" 2>/dev/null || true
  wait "$tensorboard_pid" 2>/dev/null || true
}
trap cleanup EXIT
sleep 3
kill -0 "$tensorboard_pid"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    --shm-size=32g \
    -v "$repo:$container_repo:ro" \
    -v /data:/data \
    -e "PYTHONPATH=$container_repo/src" \
    -e NCCL_DEBUG=WARN \
    -w "$container_repo" \
    "$image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_train \
      --config "$config" \
      --batch-per-gpu "$batch" \
      --max-updates 50000 \
      "${resume_args[@]}"
