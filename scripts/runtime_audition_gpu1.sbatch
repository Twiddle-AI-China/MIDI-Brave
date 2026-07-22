#!/usr/bin/env bash
#SBATCH --job-name=mb-runtime-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/runtime-audition-%j.out

set -euo pipefail
: "${CHECKPOINT:?submit with --export=ALL,CHECKPOINT=/data/.../update-N.pt}"
: "${OUTPUT:?submit with --export=ALL,OUTPUT=/data/.../runtime-audition-N}"

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
image=midibrave:v0.10.0-clap-recon
config=${CONFIG:-/workspace/MidiBrave/configs/v3/octopus_pad50_phase3.yaml}
seeds=${SEEDS:-3}
timbres=${TIMBRES:-4}
notes=${NOTES:-4}
duration=${DURATION:-2.5}
warmup=${WARMUP:-0.4}

mkdir -p "$OUTPUT"
srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w /workspace/MidiBrave "$image" \
  python scripts/render_runtime_audition.py \
  --config "$config" --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --seeds "$seeds" --timbres "$timbres" --notes "$notes" \
  --duration "$duration" --warmup "$warmup" --device cuda
