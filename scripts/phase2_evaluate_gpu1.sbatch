#!/usr/bin/env bash
#SBATCH --job-name=mb-p2-eval
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase2-eval-%j.out

set -euo pipefail
: "${CHECKPOINT:?submit with --export=ALL,CHECKPOINT=/data/.../update-N.pt}"
: "${OUTPUT:?submit with --export=ALL,OUTPUT=/data/.../evaluation-N}"

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
image=midibrave:v0.10.0-clap-recon
config=/workspace/MidiBrave/configs/v3/octopus_pad50_phase2.yaml
pairs=${PAIRS:-128}

mkdir -p "$OUTPUT"
srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w /workspace/MidiBrave "$image" \
  midibrave-evaluate --config "$config" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --pairs "$pairs" --device cuda
