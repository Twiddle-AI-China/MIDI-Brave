#!/usr/bin/env bash
#SBATCH --job-name=zrave-standalone-eval
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yfhuang/zrave-standalone-eval-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
container_repo=/workspace/MidiBrave
config=$container_repo/configs/zrave/octopus_standalone50.yaml
image=midibrave:v0.9.0-five-model-fp16
root=/srv/data-branches/nvme/midibrave-zrave
container_root=/data/midibrave-zrave-standalone
run_root=$root/runs/zrave_transformer_balanced50
container_run=$container_root/runs/zrave_transformer_balanced50
checkpoint=${ZRAVE_CHECKPOINT:-$run_root/checkpoints/best.pt}
checkpoint_name=$(basename "$checkpoint" .pt)
container_checkpoint=$container_run/checkpoints/$checkpoint_name.pt
report_root=$run_root/evaluation
container_report=$container_run/evaluation

test -f "$checkpoint"
mkdir -p "$report_root"

for split in validation test; do
  srun --kill-on-bad-exit=1 --cpu-bind=cores \
    docker run --rm \
      --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
      --shm-size=16g \
      -v "$repo:$container_repo:ro" \
      -v "$root:$container_root" \
      -e "PYTHONPATH=$container_repo/src" \
      -w "$container_repo" \
      "$image" \
      python -m midibrave.zrave_evaluate \
        --config "$config" \
        --checkpoint "$container_checkpoint" \
        --split "$split" \
        --output "$container_report/${checkpoint_name}-${split}.json" \
        --batch-size 64 \
        --device cuda \
        --allow-failed-gate
done
