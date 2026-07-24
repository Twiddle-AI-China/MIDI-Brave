#!/usr/bin/env bash
#SBATCH --job-name=zrave-flow-sweep
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yfhuang/zrave-flow-sweep-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
pitch_root=$flow_root/runs/pitch-probe
sweep_root=$host_flow_root/sweeps/${SLURM_JOB_ID}
latest=$host_flow_root/sweeps/latest-selection.json
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -f "$host_flow_root/preparation.complete"
test -f "$host_flow_root/runs/pitch-probe/qualified.complete"
test -f "$host_flow_root/runs/pitch-probe/qualification.json"
mkdir -p "$sweep_root"

for batch in 128 256 384 512; do
  report=$sweep_root/batch-${batch}.json
  set +e
  srun --kill-on-bad-exit=1 --cpu-bind=cores \
    docker run --rm \
      --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
      --shm-size=32g \
      -v "$repo_root:$container_root:ro" \
      -v "$dataset_mount" \
      -v "$codec_mount" \
      -v "$host_flow_root:$flow_root" \
      -e "PYTHONPATH=$container_repo/src" \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -e NCCL_DEBUG=WARN \
      -w "$container_repo" \
      "$image" \
      torchrun --standalone --nproc_per_node=8 \
        -m midibrave.zrave_flow_train \
        --config "$config" \
        --pitch-probe "$pitch_root" \
        --batch-per-gpu "$batch" \
        --benchmark-report "$flow_root/sweeps/${SLURM_JOB_ID}/batch-${batch}.json" \
        --benchmark-warmup 20 \
        --benchmark-updates 100 \
        --benchmark-exposure-updates 5
  process_status=$?
  set -e
  if [[ $process_status -ne 0 || ! -f "$report" ]]; then
    python3 - "$report" "$batch" "$process_status" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "failed",
    "batch_per_gpu": int(sys.argv[2]),
    "world_size": 8,
    "global_batch": int(sys.argv[2]) * 8,
    "measured_updates": 0,
    "exposure_safety_updates": 0,
    "median_windows_per_second": 0.0,
    "median_valid_latent_frames_per_second": 0.0,
    "p10_windows_per_second": 0.0,
    "peak_memory_mib": 0.0,
    "total_memory_mib": 32768.0,
    "nonfinite_updates": 0,
    "process_exit_code": int(sys.argv[3]),
}, indent=2, sort_keys=True) + "\n")
PY
  fi
done

python3 "$repo/scripts/cloud/select_zrave_sweep.py" \
  --sweep-root "$sweep_root" \
  --output "$sweep_root/selection.json" \
  --latest-output "$latest" \
  --metric median_valid_latent_frames_per_second \
  --batches 128 256 384 512
