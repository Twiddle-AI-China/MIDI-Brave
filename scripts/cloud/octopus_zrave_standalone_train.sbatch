#!/usr/bin/env bash
#SBATCH --job-name=zrave-standalone-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yfhuang/zrave-standalone-train-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
run_root=$root/runs/zrave_transformer_mean50_v2
container_run=$container_root/runs/zrave_transformer_mean50_v2
selection=${SWEEP_SELECTION:-$root/sweeps/mean-v2/latest-selection.json}

test -f "$selection"
batch=$(python3 - "$selection" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1]))["batch_per_gpu"]))
PY
)
case "$batch" in
  1024|1536|2048|2560) ;;
  *)
    echo "invalid selected batch: $batch" >&2
    exit 2
    ;;
esac

mkdir -p "$run_root/checkpoints" "$run_root/tensorboard"
resume_args=()
resume=$(find "$run_root/checkpoints" -maxdepth 1 -type f \
  -name 'update-*.pt' -printf '%f\n' | sort | tail -n 1 || true)
if [[ -n "$resume" ]]; then
  resume_args=(--resume "$container_run/checkpoints/$resume")
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
      "${resume_args[@]}"
