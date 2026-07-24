#!/usr/bin/env bash
#SBATCH --job-name=zrave-flow-prepare
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yfhuang/zrave-flow-prepare-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

test -d "$repo"
test -f "$repo/configs/zrave/octopus_flow.yaml"
test -d /srv/data-branches/nvme/datasets
test -d /data/midibrave-zrave-standalone
mkdir -p \
  "$host_flow_root/manifests" \
  "$host_flow_root/cache" \
  "$host_flow_root/packs"

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
    python - "$config" <<'PY'
import sys
from pathlib import Path
from midibrave.zrave_flow_config import ZraveFlowConfig

cfg = ZraveFlowConfig.load(sys.argv[1])
for source in cfg.data.sources:
    assert Path(source.manifest).is_file(), source.manifest
    assert Path(source.audio_root).is_dir(), source.audio_root
    if source.preset_metadata:
        assert Path(source.preset_metadata).is_file(), source.preset_metadata
assert Path(cfg.rave.checkpoint).is_file(), cfg.rave.checkpoint
PY

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
    python -m midibrave.zrave_flow_data \
      --config "$config" manifest

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
    -w "$container_repo" \
    "$image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_flow_data \
      --config "$config" encode

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
    python -m midibrave.zrave_flow_data \
      --config "$config" finalize

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
    python - "$config" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from midibrave.zrave_flow_config import ZraveFlowConfig

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

cfg = ZraveFlowConfig.load(sys.argv[1])
root = Path(cfg.data.packed_root)
index = json.loads((root / "index.json").read_text())
assert index["rave_checkpoint_sha256"] == cfg.rave.expected_sha256
assert index["latent_dim"] == 16
assert index["seed_bank_missing_notes"] == []
seed_bank = json.loads((root / "seed-bank.json").read_text())
assert seed_bank["missing_notes"] == []
assert {row["midi_note"] for row in seed_bank["entries"]} == set(range(21, 110))
for name, expected in index["shard_sha256"].items():
    assert digest(root / name) == expected, name
for name, field in (
    ("statistics.npz", "statistics_sha256"),
    ("pitch-pairs.npy", "pitch_pairs_sha256"),
    ("seed-bank.npz", "seed_bank_npz_sha256"),
    ("seed-bank.json", "seed_bank_json_sha256"),
):
    assert digest(root / name) == index[field], name
PY

touch "$host_flow_root/preparation.complete"
