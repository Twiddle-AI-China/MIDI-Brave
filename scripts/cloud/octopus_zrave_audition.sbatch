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
zrave_branch=/srv/data-branches/hdd-b/data/midibrave-zrave
v3_branch=/srv/data-branches/hdd-a/data/midibrave-v3
model_branch=/srv/data-branches/nvme/model_weights
dataset_branch=/srv/data-branches/nvme/datasets
host_output=${ZRAVE_AUDITION_OUTPUT:-/srv/data-branches/nvme/midibrave-zrave/audition/best-8000-v1}
output=/output
page=$container_repo/audition/zrave-sequence-comparison/index.html

test -d "$repo"
test -f "$repo/configs/zrave/octopus_balanced50.yaml"
test -f "$repo/scripts/render_zrave_audition.py"
test -f "$repo/audition/zrave-sequence-comparison/index.html"
test -f "$zrave_branch/runs/zrave_transformer_balanced50/checkpoints/update-00008000.pt"
test -f "$zrave_branch/packs/serum-balanced50/index.json"
test -f "$zrave_branch/packs/serum-balanced50/latents.npy"
test -f "$v3_branch/runs/pad50_scratch_control_rescue_window_clap_v1/rollout/update-00000500.pt"
test -f "$model_branch/laion-clap/music_audioset_epoch_15_esc_90.14.pt"
test -d "$dataset_branch/latent-cosmos-synth/serum-dataset"
test ! -e "$host_output"

mkdir -p "$host_output"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    --shm-size=16g \
    -v "$repo:$container_repo:ro" \
    -v "$zrave_branch:/data/midibrave-zrave:ro" \
    -v "$v3_branch:/data/midibrave-v3:ro" \
    -v "$model_branch:/data/model_weights:ro" \
    -v "$dataset_branch:/data/datasets:ro" \
    -v "$host_output:$output" \
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
    -v "$host_output:$output" \
    "$image" \
    cp "$page" "$output/index.html"

wav_count=$(find "$host_output/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "$wav_count" -ne 32 ]]; then
  echo "expected 32 audition WAV files, found $wav_count" >&2
  exit 3
fi

python3 - "$host_output" <<'PY'
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
