#!/usr/bin/env bash
#SBATCH --job-name=zrave-serum128-explore-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yfhuang/zrave-serum128-explore-train-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_serum128_exploration_flow.yaml
host_config=$repo/configs/zrave/octopus_serum128_exploration_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow-serum128
flow_root=/data/midibrave-zrave-flow-serum128
rave_mount=/data/midibrave-rave-serum-v2:/data/midibrave-rave-serum-v2:ro
source_mount=/data/datasets/Timbre_A/serum-octopus-v1:/source:ro
run_root=$host_flow_root/runs/exploration-v2
checkpoint_root=$run_root/checkpoints
selection=$host_flow_root/sweeps/exploration-v2/latest-selection.json
initial_checkpoint=${INITIAL_CHECKPOINT:-$host_flow_root/runs/pure-flow-v1/checkpoints/step-085000.pt}
max_updates=${MAX_UPDATES:-20000}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -f "$host_flow_root/preparation.complete"
test -f "$host_flow_root/packs/serum-balanced/index.json"
test -f "$host_flow_root/packs/serum-balanced/statistics.npz"
test -f "$initial_checkpoint"
if [[ ! "$max_updates" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_UPDATES must be a positive integer" >&2
  exit 2
fi
mkdir -p "$checkpoint_root" "$run_root/tensorboard"

batch=${BATCH_PER_GPU:-}
if [[ -z "$batch" ]]; then
  test -f "$selection"
  batch=$(python3 - \
    "$selection" \
    "$host_config" \
    "$host_flow_root/packs/serum-balanced/index.json" \
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

selection_path, config_path, index_path, repo = map(Path, sys.argv[1:])
selection = json.loads(selection_path.read_text())
selected = selection["selected"]
expected = {
    "config_sha256": digest(config_path),
    "pack_index_sha256": digest(index_path),
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
assert selection["metric"] == "median_valid_latent_frames_per_second"
batch = int(selection["batch_per_gpu"])
assert batch in (128, 160, 192, 224)
print(batch)
PY
  )
fi
if [[ ! "$batch" =~ ^(128|160|192|224)$ ]]; then
  echo "BATCH_PER_GPU must be 128, 160, 192, or 224" >&2
  exit 2
fi

checkpoint_args=()
resume_name=${SERUM128_EXPLORATION_RESUME_BASENAME:-}
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
  checkpoint_args=(
    --resume "$flow_root/runs/exploration-v2/checkpoints/$resume_name"
  )
else
  checkpoint_args=(--initialize-from "$initial_checkpoint")
fi

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=32g \
    -v "$repo_root:$container_root:ro" \
    -v "$rave_mount" \
    -v "$source_mount" \
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
      --batch-per-gpu "$batch" \
      --max-updates "$max_updates" \
      "${checkpoint_args[@]}"
