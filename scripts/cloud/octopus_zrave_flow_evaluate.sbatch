#!/usr/bin/env bash
#SBATCH --job-name=zrave-flow-evaluate
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yfhuang/zrave-flow-evaluate-%j.out

set -euo pipefail

repo_root=/home/yfhuang/Latent-Cosmos-Synth-midiBrave
repo=$repo_root/MidiBrave-v2
container_root=/workspace/Latent-Cosmos-Synth-midiBrave
container_repo=$container_root/MidiBrave-v2
config=$container_repo/configs/zrave/octopus_flow.yaml
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-zrave-flow}
host_flow_root=/data/midibrave-zrave-flow
flow_root=/data/midibrave-zrave-flow
run_root=$host_flow_root/runs/flow-v1
checkpoint_root=$run_root/checkpoints
dataset_mount=/srv/data-branches/nvme/datasets:/data/datasets:ro
codec_mount=/data/midibrave-zrave-standalone:/data/midibrave-zrave-standalone:ro
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${FLOW_CHECKPOINT_BASENAME:?set an explicit checkpoint basename}"
if [[ "$FLOW_CHECKPOINT_BASENAME" == */* || "$FLOW_CHECKPOINT_BASENAME" != *.pt ]]; then
  echo "checkpoint must be a basename ending in .pt" >&2
  exit 2
fi
checkpoint=$checkpoint_root/$FLOW_CHECKPOINT_BASENAME
test -f "$checkpoint"
checkpoint_stem=${FLOW_CHECKPOINT_BASENAME%.pt}
output=$run_root/evaluations/test-${checkpoint_stem}
mkdir -p "$output"

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
    python -c 'import torch; torch.use_deterministic_algorithms(True); from midibrave.zrave_flow_evaluate import main; main()' \
      --config "$config" \
      --checkpoint "$flow_root/runs/flow-v1/checkpoints/$FLOW_CHECKPOINT_BASENAME" \
      --split test \
      --output "$flow_root/runs/flow-v1/evaluations/test-${checkpoint_stem}" \
      --device cuda
