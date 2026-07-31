#!/usr/bin/env bash
#SBATCH --job-name=zrave-serum128-prepare
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yfhuang/zrave-serum128-prepare-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_serum128_pure_flow.yaml
flow_image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
rave_image=${RAVE_IMAGE:-latent-cosmos-rave:serum-v2-rate-v3}
host_flow_root=/data/midibrave-zrave-flow-serum128
flow_root=/data/midibrave-zrave-flow-serum128
rave_root=/data/midibrave-rave-serum-v2
rave_mount=/data/midibrave-rave-serum-v2:/data/midibrave-rave-serum-v2:ro
source_mount=/data/datasets/Timbre_A/serum-octopus-v1:/source:ro
source_checkpoint=$rave_root/phase2/runs/stage-a/serum_rave_v2_phase2_stage_a_9ca0f3c2a0/version_4/checkpoints/epoch-epoch=0228.ckpt
source_config=$rave_root/phase2/entry-global927261/config.gin
source_sha=3a158421567618860808d7f92d301f222b7243591a493ca53bd285cd9ad9bde7
codec=$host_flow_root/codec/serum-balanced-rave128-full.ts
codec_sha=69e8a2a133151eb896842100c993c6f4904936f08ee72e430f1c9c2556b6f9b4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -d "$repo"
test -f "$source_checkpoint"
test -f "$source_config"
test -d /data/datasets/Timbre_A/serum-octopus-v1
test "$(sha256sum "$source_checkpoint" | awk '{print $1}')" = "$source_sha"
mkdir -p \
  "$host_flow_root/codec" \
  "$host_flow_root/manifests" \
  "$host_flow_root/cache" \
  "$host_flow_root/packs"

if [[ ! -f "$codec" ]]; then
  srun --kill-on-bad-exit=1 --cpu-bind=cores \
    docker run --rm \
      --network none \
      -v "$repo_root:$container_root:ro" \
      -v "$rave_mount" \
      -v "$host_flow_root:$flow_root" \
      -e "PYTHONPATH=$container_repo/src" \
      -w "$container_repo" \
      "$rave_image" \
      python -m midibrave.rave_full_latent_export \
        --source-checkpoint "$source_checkpoint" \
        --source-config "$source_config" \
        --output "$flow_root/codec/serum-balanced-rave128-full.ts" \
        --expected-source-sha256 "$source_sha" \
        --expected-global-step 968805 \
        --latent-dim 128 \
        --sample-rate 44100 \
        --latent-hop 2048
fi
test "$(sha256sum "$codec" | awk '{print $1}')" = "$codec_sha"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --network none \
    -v "$repo_root:$container_root:ro" \
    -v "$rave_mount" \
    -v "$source_mount" \
    -v "$host_flow_root:$flow_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -w "$container_repo" \
    "$flow_image" \
    python -m midibrave.zrave_flow_data \
      --config "$config" manifest

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
    "$flow_image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_flow_data \
      --config "$config" encode

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --network none \
    -v "$repo_root:$container_root:ro" \
    -v "$rave_mount" \
    -v "$source_mount" \
    -v "$host_flow_root:$flow_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -w "$container_repo" \
    "$flow_image" \
    python -m midibrave.zrave_flow_data \
      --config "$config" finalize

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --network none \
    -v "$repo_root:$container_root:ro" \
    -v "$rave_mount" \
    -v "$source_mount" \
    -v "$host_flow_root:$flow_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -w "$container_repo" \
    "$flow_image" \
    python - "$config" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from midibrave.zrave_flow_config import ZraveFlowConfig

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

config = ZraveFlowConfig.load(sys.argv[1])
root = Path(config.data.packed_root)
index = json.loads((root / "index.json").read_text())
assert index["records"] > 100_000, index["records"]
assert index["latent_dim"] == 128
assert index["latent_hop"] == 2048
assert index["rave_checkpoint_sha256"] == config.rave.expected_sha256
with np.load(root / "statistics.npz") as statistics:
    assert statistics["mean"].shape == (128,)
    assert statistics["latent_std"].shape == (128,)
    assert np.isfinite(statistics["mean"]).all()
    assert np.isfinite(statistics["latent_std"]).all()
for name, expected in index["shard_sha256"].items():
    assert digest(root / name) == expected, name
PY

touch "$host_flow_root/preparation.complete"
