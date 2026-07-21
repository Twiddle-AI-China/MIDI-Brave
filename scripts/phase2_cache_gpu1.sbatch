#!/usr/bin/env bash
#SBATCH --job-name=mb-p2-cache
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase2-cache-%j.out

set -euo pipefail
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
launch=/data/midibrave-v3/launches/pad50-20260721
image=midibrave:v0.10.0-clap-recon
config=/workspace/MidiBrave/configs/v3/octopus_pad50_phase2.yaml
cache=/data/midibrave-v3/cache/pad50-phase15-5k
shared=/data/midibrave/cache/serum_strict_1822
checkpoint=/data/midibrave-v3/runs/pad50_predictive_rave_phase1_5_from28k/rave/update-00005000.pt

mkdir -p "$launch/logs" "$launch/monitor" "$cache"
for name in audio clap pitch; do
  if [[ -e "$cache/$name" && ! -L "$cache/$name" ]]; then
    echo "refusing to replace non-symlink cache path: $cache/$name" >&2
    exit 1
  fi
  ln -sfn "$shared/$name" "$cache/$name"
done
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 10 > "$launch/monitor/phase2-cache-${SLURM_JOB_ID}.csv" 2> "$launch/monitor/phase2-cache-${SLURM_JOB_ID}.err" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=16g \
  -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
  -e PYTHONPATH=/workspace/MidiBrave/src \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -w /workspace/MidiBrave "$image" \
  python -m midibrave.cli cache-rave \
  --config "$config" --checkpoint "$checkpoint" --device cuda

expected=$(wc -l < /data/midibrave-v3/manifests/pad50-strict.jsonl)
actual=$(find "$cache/rave" -maxdepth 1 -type f -name '*.npz' | wc -l)
[[ "$actual" -eq "$expected" ]]
sha256sum "$checkpoint" "$cache/rave-statistics.npz"
echo "phase2 cache verified: $actual/$expected records"
