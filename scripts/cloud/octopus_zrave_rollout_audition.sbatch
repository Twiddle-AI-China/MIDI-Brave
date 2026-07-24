#!/usr/bin/env bash
#SBATCH --job-name=zrave-rollout-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-rollout-audition-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50_rollout.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
run_root=$root/runs/zrave_transformer_rollout_v3
container_run=$container_root/runs/zrave_transformer_rollout_v3
checkpoint_name=${ZRAVE_CHECKPOINT_NAME:-best.pt}

if [[ "$checkpoint_name" != "$(basename "$checkpoint_name")" ]]; then
  echo "ZRAVE_CHECKPOINT_NAME must be a filename" >&2
  exit 2
fi
checkpoint=$run_root/checkpoints/$checkpoint_name
container_checkpoint=$container_run/checkpoints/$checkpoint_name
checkpoint_stem=${checkpoint_name%.pt}
audition_name=${ZRAVE_AUDITION_NAME:-standalone-rollout-v3-${checkpoint_stem}}
if [[ "$audition_name" != "$(basename "$audition_name")" ]]; then
  echo "ZRAVE_AUDITION_NAME must be one directory name" >&2
  exit 3
fi
output=$root/audition/$audition_name
container_output=$container_root/audition/$audition_name

test -f "$checkpoint"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing audition output: $output" >&2
  exit 4
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
  -e "PYTHONPATH=$container_repo/src" \
  -w "$container_repo" \
  "$image" \
  python -m midibrave.zrave_audio_quality \
    --manifest "$container_output/audition-manifest.json" \
    --output "$container_output/audio-quality.json"

docker run --rm \
  -v "$repo:$container_repo:ro" \
  -v "$root:$container_root" \
  "$image" \
  cp "$container_repo/audition/zrave-sequence-comparison/index.html" \
    "$container_output/index.html"

wav_count=$(find "$output/audio" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "$wav_count" -ne 32 ]]; then
  echo "expected 32 audition WAV files, got $wav_count" >&2
  exit 5
fi
test -f "$output/audio-quality.json"
