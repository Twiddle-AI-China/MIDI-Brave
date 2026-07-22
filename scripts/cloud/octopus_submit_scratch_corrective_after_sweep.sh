#!/usr/bin/env bash
set -euo pipefail

sweep_job=${1:?usage: octopus_submit_scratch_corrective_after_sweep.sh SWEEP_JOB_ID}
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
summary=/data/midibrave-v3/sweeps/scratch-pad-corrective-${sweep_job}/summary.tsv

test -s "$summary"
winner=$(awk -F '\t' '
  NR > 1 && $2 == 0 && $4 != "NA" && ($5 + 0) < 15360 {
    throughput = $4 + 0
    if (!found || throughput > best) {
      found = 1
      best = throughput
      batch = $1
    }
  }
  END {
    if (!found) exit 1
    print batch
  }
' "$summary")

echo "sweep_job=$sweep_job winner_batch_per_gpu=$winner"
sbatch --export=ALL,BATCH_PER_GPU="$winner",MAX_STEPS=100000 \
  "$repo/scripts/cloud/octopus_scratch_rollout_corrective_train.sbatch"
