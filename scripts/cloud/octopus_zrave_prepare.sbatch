#!/usr/bin/env bash
#SBATCH --job-name=zrave-prepare
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yfhuang/zrave-prepare-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_balanced50.yaml
image=midibrave:v0.10.0-clap-recon

test -d "$repo"
test -f "$repo/configs/zrave/octopus_balanced50.yaml"
test -f /data/midibrave-v3/runs/pad50_scratch_control_rescue_window_clap_v1/rollout/update-00000500.pt
mkdir -p \
  /data/midibrave-zrave/manifests \
  /data/midibrave-zrave/cache/serum-balanced50 \
  /data/midibrave-zrave/packs/serum-balanced50

docker run --rm \
  -v "$repo:$container_repo:ro" \
  -v /data:/data \
  -e "PYTHONPATH=$container_repo/src" \
  -w "$container_repo" \
  "$image" \
  python -m midibrave.zrave_data select --config "$config"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=32g \
    -v "$repo:$container_repo:ro" \
    -v /data:/data \
    -e "PYTHONPATH=$container_repo/src" \
    -e WORLD_SIZE=8 \
    -w "$container_repo" \
    "$image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_data cache-and-pack --config "$config"

python3 - <<'PY'
import json
from pathlib import Path

index = json.loads(Path(
    "/data/midibrave-zrave/packs/serum-balanced50/index.json"
).read_text())
assert index["records"] == 3600, index["records"]
assert index["latent_dim"] == 16, index["latent_dim"]
assert index["split_records"] == {
    "train": 2880,
    "validation": 360,
    "test": 360,
}, index["split_records"]
print(json.dumps({
    "event": "zrave_prepare_complete",
    "records": index["records"],
    "rave_checkpoint_sha256": index["rave_checkpoint_sha256"],
    "latents_sha256": index["latents_sha256"],
}, sort_keys=True))
PY
