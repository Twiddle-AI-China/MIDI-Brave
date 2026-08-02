#!/usr/bin/env bash
#SBATCH --job-name=zrave-serum128-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-serum128-audition-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_serum128_pure_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
flow_root=/data/midibrave-zrave-flow-serum128
checkpoint_root=$flow_root/runs/pure-flow-v1/checkpoints
audition_root=$flow_root/auditions
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${SERUM128_FLOW_AUDITION_CHECKPOINT:?set an explicit checkpoint basename}"
if [[ "$SERUM128_FLOW_AUDITION_CHECKPOINT" == */* \
  || "$SERUM128_FLOW_AUDITION_CHECKPOINT" != *.pt ]]; then
  echo "checkpoint must be a basename ending in .pt" >&2
  exit 2
fi
checkpoint=$checkpoint_root/$SERUM128_FLOW_AUDITION_CHECKPOINT
checkpoint_stem=${SERUM128_FLOW_AUDITION_CHECKPOINT%.pt}
audition_name=${SERUM128_FLOW_AUDITION_NAME:-serum128-flow-${checkpoint_stem}-long320}
output=$audition_root/$audition_name

test -f "$checkpoint"
test -f "$flow_root/packs/serum-balanced/index.json"
test -f "$flow_root/codec/serum-balanced-rave128-full.ts"
if [[ -e "$output" ]]; then
  echo "audition output already exists: $output" >&2
  exit 2
fi
mkdir -p "$output"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=16g \
    -v "$repo_root:$container_root:ro" \
    -v "/data/midibrave-zrave-flow-serum128:/data/midibrave-zrave-flow-serum128:ro" \
    -v "/data/datasets/Timbre_A/serum-octopus-v1:/source:ro" \
    -v "$output:/output" \
    -e "PYTHONPATH=$container_repo/src" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -w "$container_repo" \
    "$image" \
    python scripts/render_zrave_flow_audition.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output /output \
      --categories Pad Lead Bass Pluck Keys Synth \
      --temperatures 0.7 1.0 \
      --generation-seeds 17 71 \
      --generated-frames 320 \
      --selection-seed 20260802 \
      --device cuda

docker run --rm -i \
  -v "$output:/output" \
  "$image" \
  python - /output <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["schema"] == 1
assert manifest["codec"]["latent_dim"] == 128
assert manifest["context_frames"] == 32
assert manifest["generated_frames"] == 320
assert len(manifest["examples"]) == 6
paths = []
for example in manifest["examples"]:
    assert len(example["rollouts"]) == 4
    for take in [example["source"], example["direct"], *example["rollouts"]]:
        paths.extend((take["raw_wav"], take["matched_wav"]))
assert len(paths) == 72
for relative in paths:
    path = root / relative
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 44100, path
    assert audio.shape[0] > 0 and audio.shape[1] == 1, path
    assert np.isfinite(audio).all(), path
report = {
    "schema": 1,
    "checkpoint_update": manifest["checkpoint"]["update"],
    "wav_files": len(paths),
    "all_finite": True,
    "sample_rate": 44100,
}
(root / "verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, sort_keys=True))
PY

tar -C "$audition_root" -czf "$output.tar.gz" "$audition_name"
printf 'audition=%s\narchive=%s\n' "$output" "$output.tar.gz"
