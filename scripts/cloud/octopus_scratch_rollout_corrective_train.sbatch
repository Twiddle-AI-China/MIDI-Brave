#!/usr/bin/env bash
#SBATCH --job-name=mb-scratch-corrective
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=0
#SBATCH --requeue
#SBATCH --output=/data/midibrave-v3/launches/scratch-pad-corrective-20260722/logs/train-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
launch=/data/midibrave-v3/launches/scratch-pad-corrective-20260722
image=midibrave:v0.10.0-clap-recon
config=/workspace/MidiBrave/configs/v3/octopus_scratch_pad_rollout_corrective.yaml
checkpoint=/data/midibrave-v3/scratch-pad-final-audition/input/update-00001500.pt
statistics=/data/midibrave-v3/cache/scratch-pad-20260722/rave-statistics.npz
calibration=/data/midibrave-v3/cache/scratch-pad-20260722/predictive-calibration.json
run_root=/data/midibrave-v3/runs/pad50_scratch_rollout_corrective_v100_8gpu/rollout
clap_failure_root=/data/midibrave-v3/diagnostics/scratch-pad-corrective-clap
batch_per_gpu=${BATCH_PER_GPU:-40}
max_steps=${MAX_STEPS:-100000}

mkdir -p "$launch/logs" "$launch/monitor" "$run_root" "$clap_failure_root"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 10 \
  > "$launch/monitor/train-${SLURM_JOB_ID}.csv" \
  2> "$launch/monitor/train-${SLURM_JOB_ID}.err" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

start_args=(--warm-start "$checkpoint")
latest=$(find "$run_root" -maxdepth 1 -type f -name 'update-*.pt' | sort | tail -n 1 || true)
if [[ -n "$latest" ]]; then
  start_args=(--resume "$latest" --allow-world-size-change)
fi

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=32g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e MIDIBRAVE_CLAP_FAILURE_DIR="$clap_failure_root" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -w /workspace/MidiBrave "$image" \
  torchrun --standalone --nproc_per_node=8 -m midibrave.trainer \
  --config "$config" --stage rollout --max-steps "$max_steps" \
  --batch-per-gpu "$batch_per_gpu" --latent-statistics "$statistics" \
  --calibration "$calibration" "${start_args[@]}"
