#!/usr/bin/env bash
#SBATCH --job-name=mb-rave-eval
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/rave-eval-%j.out
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: sbatch $0 CONFIG CHECKPOINT OUTPUT" >&2
  exit 2
fi

config=$1
checkpoint=$2
output=$3
repo=${MIDIBRAVE_REPO:-/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2}
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-clap-recon}

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" --shm-size=16g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w /workspace/MidiBrave "$image" \
  python -m midibrave.evaluate \
    --config "$config" --checkpoint "$checkpoint" --output "$output" \
    --pairs 32 --batch-size 2 --examples 8 --device cuda
