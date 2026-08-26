#!/bin/bash
set -euo pipefail

job_id=${1:?usage: spark_pause_job.sh JOB_ID}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "JOB_ID must be numeric" >&2; exit 64; }
pause_marker=/data/atlas-flow-pad-v1/contracts/spark-gb10/pause-requested
mkdir -p "$(dirname "$pause_marker")"
touch "$pause_marker"

container=$(docker ps --filter "name=atlas-flow-spark-${job_id}-" --format '{{.Names}}' | head -n 1)
if [[ -n "$container" ]]; then
  docker kill --signal=USR1 "$container" >/dev/null
  echo "checkpoint signal sent to $container"
else
  echo "no running training container found; cancelling queued job $job_id"
  scancel "$job_id"
fi
