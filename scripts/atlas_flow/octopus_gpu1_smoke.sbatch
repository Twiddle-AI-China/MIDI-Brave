#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --job-name=atlas-flow-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=00:20:00
#SBATCH --output=/data/atlas-flow-pad-v1/logs/gpu1-smoke-%j.out

set -euo pipefail
IMAGE=midibrave:atlas-flow-v1
CODE=/data/projects/latent-cosmos-synth/atlas-flow-6d143b2-20260822/MidiBrave-v2
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 70
[[ $(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES") -eq 1 ]] || exit 70
docker run --rm --network none --shm-size 4g --cpus 8 --memory 14g --memory-swap 14g \
  --pids-limit 2048 --user "$(id -u):$(id -g)" \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
  --mount "type=bind,src=$CODE,dst=/code,readonly" \
  "$IMAGE" python /code/scripts/atlas_flow/gpu_model_smoke.py \
  --config /opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml \
  --report "/data/atlas-flow-pad-v1/contracts/gpu-smoke-${SLURM_JOB_ID}.json"
