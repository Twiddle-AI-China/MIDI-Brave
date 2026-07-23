#!/usr/bin/env bash
#SBATCH --job-name=zrave-eval
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yfhuang/zrave-eval-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_balanced50.yaml
image=midibrave:v0.10.0-clap-recon
run_root=/data/midibrave-zrave/runs/zrave_transformer_balanced50
checkpoint=${ZRAVE_CHECKPOINT:-$run_root/checkpoints/best.pt}
report_root=$run_root/evaluation

test -f "$checkpoint"
mkdir -p "$report_root"
checkpoint_name=$(basename "$checkpoint" .pt)

for split in validation test; do
  srun --kill-on-bad-exit=1 --cpu-bind=cores \
    docker run --rm \
      --gpus "device=${CUDA_VISIBLE_DEVICES}" \
      --shm-size=16g \
      -v "$repo:$container_repo:ro" \
      -v /data:/data \
      -e "PYTHONPATH=$container_repo/src" \
      -w "$container_repo" \
      "$image" \
      python -m midibrave.zrave_evaluate \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --split "$split" \
        --output "$report_root/${checkpoint_name}-${split}.json" \
        --batch-size 64 \
        --device cuda \
        --allow-failed-gate
done
