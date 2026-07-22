#!/usr/bin/env bash
#SBATCH --job-name=mb-pad-scratch6
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --time=04:30:00
#SBATCH --output=/data/run01/scwc257/latent-cosmos-synth/midibrave-v3/scratch-pad-20260722/logs/reservation-%j.out

set -Eeuo pipefail

ROOT=/data/run01/scwc257/latent-cosmos-synth/midibrave-v3/scratch-pad-20260722
LAUNCH="$ROOT/launch.sh"
mkdir -p "$ROOT/logs"

source /usr/share/modules/init/bash
module use /data/apps/modulefiles
module load cuda/12.8
nvidia-smi -L

while [[ ! -s "$LAUNCH" ]]; do
  sleep 5
done

exec bash "$LAUNCH"
