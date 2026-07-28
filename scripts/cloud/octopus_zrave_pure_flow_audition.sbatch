#!/usr/bin/env bash
#SBATCH --job-name=zrave-pure-flow-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-pure-flow-audition-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_pure_flow_poc.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
run_root=$host_flow_root/runs/pure-flow-poc-v1
checkpoint_root=$run_root/checkpoints
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT:?set an explicit checkpoint basename}"
if [[ "$ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT" == */* \
  || "$ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT" != *.pt ]]; then
  echo "checkpoint must be a basename ending in .pt" >&2
  exit 2
fi
checkpoint=$checkpoint_root/$ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT
checkpoint_stem=${ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT%.pt}
output=$run_root/auditions/$checkpoint_stem
test -f "$checkpoint"
if [[ -e "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "audition output already exists and is not empty: $output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"

srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --shm-size=16g \
    -v "$repo_root:$container_root:ro" \
    -v "$dataset_mount" \
    -v "$codec_mount" \
    -v "$host_flow_root:$flow_root" \
    -e "PYTHONPATH=$container_repo/src" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -w "$container_repo" \
    "$image" \
    python -m midibrave.zrave_pure_flow_audition \
      --config "$config" \
      --checkpoint "$flow_root/runs/pure-flow-poc-v1/checkpoints/$ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT" \
      --output "$flow_root/runs/pure-flow-poc-v1/auditions/$checkpoint_stem" \
      --device cuda
