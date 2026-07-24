#!/usr/bin/env bash
#SBATCH --job-name=zrave-long320
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-long320-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50_rollout.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
run_root=$root/runs/zrave_transformer_rollout_v3
container_run=$container_root/runs/zrave_transformer_rollout_v3
checkpoint_name=${ZRAVE_CHECKPOINT_NAME:-update-00002250.pt}
audition_name=${ZRAVE_LONG_AUDITION_NAME:-standalone-rollout-u2250-long320}

if [[ "$checkpoint_name" != "$(basename "$checkpoint_name")" ]]; then
  echo "ZRAVE_CHECKPOINT_NAME must be a filename" >&2
  exit 2
fi
if [[ "$audition_name" != "$(basename "$audition_name")" ]]; then
  echo "ZRAVE_LONG_AUDITION_NAME must be one directory name" >&2
  exit 3
fi

checkpoint=$run_root/checkpoints/$checkpoint_name
container_checkpoint=$container_run/checkpoints/$checkpoint_name
output=$root/audition/$audition_name
container_output=$container_root/audition/$audition_name

test -f "$checkpoint"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing long rollout: $output" >&2
  exit 4
fi

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=16g \
    -v "$repo:$container_repo:ro" \
    -v "$root:$container_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -w "$container_repo" \
    "$image" \
    python scripts/render_zrave_long_rollout.py \
      --config "$config" \
      --checkpoint "$container_checkpoint" \
      --output "$container_output" \
      --seed-frames 32 \
      --predicted-frames 320 \
      --random-seed 20260724 \
      --device cuda

docker run --rm \
  -v "$repo:$container_repo:ro" \
  -v "$root:$container_root" \
  "$image" \
  cp "$container_repo/audition/zrave-long-rollout/index.html" \
    "$container_output/index.html"

manifest=$output/long-rollout-manifest.json
test -f "$manifest"
wav_count=$(find "$output/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "$wav_count" -ne 5 ]]; then
  echo "expected 5 long-rollout WAV files, got $wav_count" >&2
  exit 5
fi
python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["schema"] == 1
assert manifest["conditioning"] == []
assert manifest["seed_frames"] == 32
assert manifest["predicted_frames"] == 320
assert manifest["rollout_calls"] == 40
assert manifest["transformer"]["update"] == 2250
assert len(manifest["rows"]) == 5
assert all(row["sample_count"] == 720896 for row in manifest["rows"])
PY
