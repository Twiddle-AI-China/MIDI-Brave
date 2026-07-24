#!/usr/bin/env bash
#SBATCH --job-name=zrave-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yfhuang/zrave-audition-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_balanced50.yaml
image=midibrave:v0.10.0-clap-recon
checkpoint=/data/midibrave-zrave/runs/zrave_transformer_balanced50/checkpoints/update-00008000.pt
output=${ZRAVE_AUDITION_OUTPUT:-/data/midibrave-zrave/audition/best-8000-v1}
page=$container_repo/audition/zrave-sequence-comparison/index.html

test -d "$repo"
test -f "$repo/configs/zrave/octopus_balanced50.yaml"
test -f "$repo/scripts/render_zrave_audition.py"
test -f "$repo/audition/zrave-sequence-comparison/index.html"
test -f "$checkpoint"
test -f /data/midibrave-zrave/packs/serum-balanced50/index.json
test -f /data/midibrave-zrave/packs/serum-balanced50/latents.npy
test -f /data/model_weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt
test ! -e "$output"

mkdir -p "$(dirname "$output")"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    --shm-size=16g \
    -v "$repo:$container_repo:ro" \
    -v /data:/data \
    -e "PYTHONPATH=$container_repo/src" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -w "$container_repo" \
    "$image" \
    python scripts/render_zrave_audition.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$output" \
      --duration 2.5 \
      --random-seed 20260724 \
      --device cuda

srun --kill-on-bad-exit=1 \
  docker run --rm \
    -v "$repo:$container_repo:ro" \
    -v /data:/data \
    "$image" \
    cp "$page" "$output/index.html"

wav_count=$(find "$output/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "$wav_count" -ne 32 ]]; then
  echo "expected 32 audition WAV files, found $wav_count" >&2
  exit 3
fi

python3 - "$output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "audition-manifest.json").read_text())
assert manifest["schema"] == 1
assert manifest["categories"] == ["Pad", "Bass", "Lead", "Pluck", "Keys"]
assert len(manifest["comparisons"]) == 10
assert len(manifest["random_continuations"]) == 2
assert manifest["models"]["transformer"]["update"] == 8000
referenced = {
    item
    for row in manifest["comparisons"]
    for item in row["audio"].values()
}
referenced.update(row["file"] for row in manifest["random_continuations"])
assert len(referenced) == 32
assert all((root / item).is_file() for item in referenced)
assert (root / "index.html").is_file()
print(json.dumps({
    "event": "zrave_audition_complete",
    "comparisons": len(manifest["comparisons"]),
    "random_continuations": len(manifest["random_continuations"]),
    "wav_files": len(referenced),
    "output": str(root),
}, sort_keys=True))
PY
