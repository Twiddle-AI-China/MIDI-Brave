#!/usr/bin/env bash
#SBATCH --job-name=zrave-sweep
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yfhuang/zrave-sweep-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_balanced50.yaml
image=midibrave:v0.10.0-clap-recon
sweep_root=/data/midibrave-zrave/sweeps/${SLURM_JOB_ID}
latest=/data/midibrave-zrave/sweeps/latest-selection.json

test -f /data/midibrave-zrave/packs/serum-balanced50/index.json
mkdir -p "$sweep_root"

for batch in 128 256 384 512; do
  report=$sweep_root/batch-${batch}.json
  set +e
  srun --kill-on-bad-exit=1 --cpu-bind=cores \
    docker run --rm \
      --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
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
        --benchmark-warmup 20 \
        --benchmark-updates 100 \
        --benchmark-output "$report"
  status=$?
  set -e
  if [[ $status -ne 0 || ! -f "$report" ]]; then
    python3 - "$report" "$batch" "$status" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
batch = int(sys.argv[2])
status = int(sys.argv[3])
path.write_text(json.dumps({
    "batch_per_gpu": batch,
    "world_size": 8,
    "global_batch": batch * 8,
    "measured_updates": 0,
    "median_windows_per_second": 0.0,
    "p10_windows_per_second": 0.0,
    "peak_memory_mib": 0.0,
    "total_memory_mib": 32768.0,
    "nonfinite_updates": 0,
    "status": "failed",
    "process_exit_code": status,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
done

python3 "$repo/scripts/cloud/select_zrave_sweep.py" \
  --sweep-root "$sweep_root" \
  --output "$sweep_root/selection.json" \
  --latest-output "$latest"
