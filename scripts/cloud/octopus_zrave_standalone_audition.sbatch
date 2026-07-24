#!/usr/bin/env bash
#SBATCH --job-name=zrave-standalone-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-standalone-audition-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
checkpoint=$root/runs/zrave_transformer_mean50_v2/checkpoints/best.pt
container_checkpoint=$container_root/runs/zrave_transformer_mean50_v2/checkpoints/best.pt
output=$root/audition/standalone-mean-v2-best-2750
container_output=$container_root/audition/standalone-mean-v2-best-2750

test -f "$checkpoint"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing audition output: $output" >&2
  exit 2
fi

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=16g \
    -v "$repo:$container_repo:ro" \
    -v "$root:$container_root" \
    -v /srv/data-branches/nvme/datasets:/data/datasets:ro \
    -e "PYTHONPATH=$container_repo/src" \
    -w "$container_repo" \
    "$image" \
    python scripts/render_zrave_audition.py \
      --config "$config" \
      --checkpoint "$container_checkpoint" \
      --output "$container_output" \
      --duration 2.5 \
      --random-seed 20260724 \
      --device cuda

docker run --rm \
  -v "$repo:$container_repo:ro" \
  -v "$root:$container_root" \
  "$image" \
  cp "$container_repo/audition/zrave-sequence-comparison/index.html" \
    "$container_output/index.html"
wav_count=$(find "$output/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "$wav_count" -ne 32 ]]; then
  echo "expected 32 audition WAV files, got $wav_count" >&2
  exit 3
fi

python3 - "$output/audition-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["conditioning"] == []
assert manifest["codec"]["kind"] == "standalone_torchscript_rave"
assert len(manifest["comparisons"]) == 10
assert len(manifest["random_continuations"]) == 2
print(json.dumps({
    "event": "zrave_standalone_audition_complete",
    "transformer_update": manifest["transformer"]["update"],
    "codec_sha256": manifest["codec"]["sha256"],
    "wav_count": 32,
}, sort_keys=True))
PY
