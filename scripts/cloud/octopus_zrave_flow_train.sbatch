#!/usr/bin/env bash
#SBATCH --job-name=zrave-flow-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/yfhuang/zrave-flow-train-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_flow.yaml
host_config=$repo/configs/zrave/octopus_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
pitch_root=$flow_root/runs/pitch-probe
run_root=$host_flow_root/runs/flow-v1
checkpoint_root=$run_root/checkpoints
selection=$host_flow_root/sweeps/latest-selection.json
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -f "$selection"
test -f "$host_flow_root/runs/pitch-probe/qualification.json"
mkdir -p "$checkpoint_root" "$run_root/tensorboard"

readarray -t launch < <(python3 - \
  "$selection" \
  "$host_config" \
  "$host_flow_root/packs/all-synth/index.json" \
  "$host_flow_root/runs/pitch-probe/qualification.json" \
  "$repo" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

selection_path, config_path, index_path, qualification_path, repo = map(
    Path, sys.argv[1:]
)
selection = json.loads(selection_path.read_text())
selected = selection["selected"]
qualification = json.loads(qualification_path.read_text())
checkpoint = qualification_path.parent / qualification["checkpoint"]
expected = {
    "config_sha256": digest(config_path),
    "pack_index_sha256": digest(index_path),
    "pitch_checkpoint_sha256": digest(checkpoint),
    "git_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
}
for name, value in expected.items():
    assert selected[name] == value, (name, selected[name], value)
batch = int(selection["batch_per_gpu"])
assert batch in (128, 256, 384, 512)
assert selection["metric"] == "median_valid_latent_frames_per_second"
assert qualification["passed"] is True
print(batch)
print(checkpoint.name)
PY
)
batch=${launch[0]}
pitch_checkpoint_name=${launch[1]}
test -f "$host_flow_root/runs/pitch-probe/checkpoints/$pitch_checkpoint_name"

resume_args=()
resume_name=${FLOW_RESUME_BASENAME:-}
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
  resume_args=(--resume "$flow_root/runs/flow-v1/checkpoints/$resume_name")
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
      -m midibrave.zrave_flow_train \
      --config "$config" \
      --pitch-probe "$pitch_root" \
      --batch-per-gpu "$batch" \
      "${resume_args[@]}"
