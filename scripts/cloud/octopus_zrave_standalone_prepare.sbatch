#!/usr/bin/env bash
#SBATCH --job-name=zrave-standalone-prepare
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yfhuang/zrave-standalone-prepare-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone

test -d "$repo"
test -f "$repo/configs/zrave/octopus_standalone50.yaml"
test -f "$root/models/freesoundloop10k_raspi_b2048_r44100_z16.ts"
mkdir -p \
  "$root/manifests" \
  "$root/cache/serum-balanced50-mean-v2" \
  "$root/packs/serum-balanced50-mean-v2"

docker run --rm \
  -v "$repo:$container_repo:ro" \
  -v "$root:$container_root" \
  -v /srv/data-branches/nvme/datasets:/data/datasets:ro \
  -v /srv/data-branches/nvme/midibrave:/data/midibrave:ro \
  -e "PYTHONPATH=$container_repo/src" \
  -w "$container_repo" \
  "$image" \
  python -m midibrave.zrave_data select --config "$config"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=32g \
    -v "$repo:$container_repo:ro" \
    -v "$root:$container_root" \
    -v /srv/data-branches/nvme/datasets:/data/datasets:ro \
    -v /srv/data-branches/nvme/midibrave:/data/midibrave:ro \
    -e "PYTHONPATH=$container_repo/src" \
    -e WORLD_SIZE=8 \
    -w "$container_repo" \
    "$image" \
    torchrun --standalone --nproc_per_node=8 \
      -m midibrave.zrave_data cache-and-pack --config "$config"

python3 - "$root/packs/serum-balanced50-mean-v2/index.json" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text())
assert index["records"] == 3600, index["records"]
assert index["latent_dim"] == 16, index["latent_dim"]
assert index["latent_hop"] == 2048, index["latent_hop"]
assert index["rave_codec"] == "standalone_torchscript_rave"
assert index["rave_latent_encoding"] == "posterior_mean_temp0_reset_v1"
assert index["streaming_state_reset_per_clip"] is True
assert index["rave_checkpoint_sha256"] == (
    "3ec093e132ce75d7fee3b8b734c739ebf8711a57ee332a60bce4359e2e34073e"
)
print(json.dumps({
    "event": "zrave_standalone_prepare_complete",
    "records": index["records"],
    "maximum_length": index["maximum_length"],
    "rave_latent_encoding": index["rave_latent_encoding"],
    "rave_checkpoint_sha256": index["rave_checkpoint_sha256"],
    "latents_sha256": index["latents_sha256"],
}, sort_keys=True))
PY
