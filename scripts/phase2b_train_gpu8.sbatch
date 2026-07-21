#!/usr/bin/env bash
#SBATCH --job-name=mb-p2b-4k
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase2b-train-%j.out

set -euo pipefail
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
launch=/data/midibrave-v3/launches/pad50-20260721
image=midibrave:v0.10.0-clap-recon
config=/workspace/MidiBrave/configs/v3/octopus_pad50_phase2b.yaml
checkpoint=/data/midibrave-v3/runs/pad50_phase2_independent_controls_from_phase15_5k/predictor/update-00001000.pt
statistics=/data/midibrave-v3/cache/pad50-phase15-5k/rave-statistics.npz
calibration=/data/midibrave-v3/cache/pad50-phase15-5k/predictive-calibration-phase2b.json
run_root=/data/midibrave-v3/runs/pad50_phase2b_stable_controls_from_p2_1k/predictor

mkdir -p "$launch/logs" "$launch/monitor" "$run_root"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 10 > "$launch/monitor/phase2b-train-${SLURM_JOB_ID}.csv" 2> "$launch/monitor/phase2b-train-${SLURM_JOB_ID}.err" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

resume_args=(--warm-start "$checkpoint")
latest=$(find "$run_root" -maxdepth 1 -type f -name 'update-*.pt' | sort | tail -n 1 || true)
if [[ -n "$latest" ]]; then
  resume_args=(--resume "$latest" --allow-world-size-change)
fi
srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=32g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -w /workspace/MidiBrave "$image" \
  torchrun --standalone --nproc_per_node=8 -m midibrave.trainer \
  --config "$config" --stage predictor \
  --latent-statistics "$statistics" --calibration "$calibration" \
  "${resume_args[@]}"
