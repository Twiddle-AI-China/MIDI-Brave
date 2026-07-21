#!/usr/bin/env bash
#SBATCH --job-name=mb-predictive-8g
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/predictive-8g-%j.out
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: sbatch $0 CONFIG STAGE [MAX_STEPS]" >&2
  exit 2
fi

config=$1
stage=$2
max_steps=${3:-}
repo=${MIDIBRAVE_REPO:-/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2}
image=${MIDIBRAVE_IMAGE:-midibrave:v0.10.0-clap-recon}
launch=${MIDIBRAVE_LAUNCH:-/data/midibrave-v3/launches/pad50-20260721}
output_dir=$(awk '$1 == "output_dir:" {print $2; exit}' "$config")
run_name=$(awk '$1 == "run_name:" {print $2; exit}' "$config")
run_root="$output_dir/$run_name/$stage"
mkdir -p "$run_root" "$launch/monitor"

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 10 > "$launch/monitor/predictive-8g-${SLURM_JOB_ID}.csv" \
  2> "$launch/monitor/predictive-8g-${SLURM_JOB_ID}.err" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

resume_args=()
latest=$(find "$run_root" -maxdepth 1 -type f -name 'update-*.pt' | sort | tail -n 1 || true)
if [[ -n "$latest" ]]; then
  resume_args=(--resume "$latest")
fi
limit_args=()
if [[ -n "$max_steps" ]]; then
  limit_args=(--max-steps "$max_steps")
fi

echo "=== PREDICTIVE TRAIN stage=$stage resume=${latest:-none} max_steps=${max_steps:-configured} ==="
srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=24g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -w /workspace/MidiBrave "$image" \
  torchrun --standalone --nproc_per_node=8 -m midibrave.trainer \
    --config "$config" --stage "$stage" "${resume_args[@]}" "${limit_args[@]}"
