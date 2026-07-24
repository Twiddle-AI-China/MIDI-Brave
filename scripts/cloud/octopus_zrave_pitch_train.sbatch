#!/usr/bin/env bash
#SBATCH --job-name=zrave-pitch-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yfhuang/zrave-pitch-train-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
pitch_root=$host_flow_root/runs/pitch-probe
checkpoint_root=$pitch_root/checkpoints
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -f "$host_flow_root/preparation.complete"
test -f "$host_flow_root/packs/all-synth/index.json"
mkdir -p "$checkpoint_root"

resume_args=()
resume_name=${PITCH_RESUME_BASENAME:-}
if [[ -z "$resume_name" ]]; then
  resume_name=$(find "$checkpoint_root" -maxdepth 1 -type f \
    -name 'step-*.pt' -printf '%f\n' | sort | tail -n 1 || true)
fi
if [[ -n "$resume_name" ]]; then
  if [[ "$resume_name" == */* || "$resume_name" != *.pt ]]; then
    echo "resume must be a checkpoint basename" >&2
    exit 2
  fi
  test -f "$checkpoint_root/$resume_name"
  resume_args=(--resume "$flow_root/runs/pitch-probe/checkpoints/$resume_name")
fi

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
      -m midibrave.zrave_pitch_train \
      --config "$config" \
      --batch-per-gpu 512 \
      --max-updates 10000 \
      "${resume_args[@]}"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    -v "$repo_root:$container_root:ro" \
    -v "$dataset_mount" \
    -v "$codec_mount" \
    -v "$host_flow_root:$flow_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -w "$container_repo" \
    "$image" \
    python - <<'PY'
import json
from pathlib import Path

root = Path("/data/midibrave-zrave-flow/runs/pitch-probe")
qualification = json.loads((root / "qualification.json").read_text())
assert qualification["passed"] is True
checkpoint = root / qualification["checkpoint"]
assert checkpoint.is_file()
PY

touch "$pitch_root/qualified.complete"
