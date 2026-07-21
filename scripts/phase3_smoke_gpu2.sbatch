#!/usr/bin/env bash
#SBATCH --job-name=mb-p3-smoke
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase3-smoke-%j.out

set -euo pipefail
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
image=midibrave:v0.10.0-clap-recon
template="$repo/configs/v3/octopus_pad50_phase3_sweep.yaml"
checkpoint=/data/midibrave-v3/runs/pad50_phase2c_control_recovery_from_p2b_3k/predictor/update-00001000.pt
statistics=/data/midibrave-v3/cache/pad50-phase15-5k/rave-statistics.npz
calibration=/data/midibrave-v3/cache/pad50-phase15-5k/predictive-calibration-phase2b.json
smoke_root="/data/midibrave-v3/smoke/phase3-${SLURM_JOB_ID}"
config="$smoke_root/config.yaml"
run_name="pad50_phase3_smoke_${SLURM_JOB_ID}"

mkdir -p "$smoke_root"
sed "s/run_name: pad50_phase3_batch_sweep/run_name: ${run_name}/" \
  "$template" > "$config"

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w /workspace/MidiBrave "$image" \
  python -m pytest tests/test_predictive_training.py tests/test_predictive_config.py -q

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g --ulimit memlock=-1 \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -w /workspace/MidiBrave "$image" \
  torchrun --standalone --nproc_per_node=2 -m midibrave.trainer \
  --config "$config" --stage rollout --max-steps 2 --batch-per-gpu 4 \
  --latent-statistics "$statistics" --calibration "$calibration" \
  --warm-start "$checkpoint"

test -s "/data/midibrave-v3/runs/${run_name}/rollout/update-00000002.pt"
