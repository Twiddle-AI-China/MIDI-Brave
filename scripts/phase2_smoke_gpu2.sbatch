#!/usr/bin/env bash
#SBATCH --job-name=mb-p2-smoke
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase2-smoke-%j.out

set -euo pipefail
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
launch=/data/midibrave-v3/launches/pad50-20260721
image=midibrave:v0.10.0-clap-recon
config=/workspace/MidiBrave/configs/v3/octopus_pad50_phase2_smoke.yaml
checkpoint=/data/midibrave-v3/runs/pad50_predictive_rave_phase1_5_from28k/rave/update-00005000.pt
statistics=/data/midibrave-v3/cache/pad50-phase15-5k/rave-statistics.npz
calibration=/data/midibrave-v3/cache/pad50-phase15-5k/predictive-calibration-smoke.json
result=/data/midibrave-v3/runs/pad50_phase2_independent_controls_smoke/predictor/update-00000002.pt

mkdir -p "$launch/logs" "$launch/monitor"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 5 > "$launch/monitor/phase2-smoke-${SLURM_JOB_ID}.csv" 2> "$launch/monitor/phase2-smoke-${SLURM_JOB_ID}.err" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -w /workspace/MidiBrave "$image" \
  python -m pytest tests/test_crepe_gpu.py -q

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=24g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -w /workspace/MidiBrave "$image" \
  torchrun --standalone --nproc_per_node=2 -m midibrave.trainer \
  --config "$config" --stage predictor --max-steps 2 \
  --warm-start "$checkpoint" --latent-statistics "$statistics" \
  --calibration "$calibration"

docker run --rm -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -w /workspace/MidiBrave "$image" \
  python scripts/verify_phase2_predictor.py \
  --warm-start "$checkpoint" --checkpoint "$result" \
  --statistics "$statistics" --expected-update 2
