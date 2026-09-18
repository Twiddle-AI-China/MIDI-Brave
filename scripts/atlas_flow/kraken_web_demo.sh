#!/usr/bin/env bash
# Atlas Flow web demo on Kraken (8x V100). Submit with:
#   sbatch scripts/atlas_flow/kraken_web_demo.sh
#
# Kraken differs from Spark in three ways that matter here:
#   * GPU containers must be started by kraken-gpu-run from inside the
#     allocation; plain `docker run --gpus` is not permitted.
#   * That wrapper creates the container on Docker's default bridge with no
#     port publishing, so the server binds 0.0.0.0 and callers reach it at the
#     container's bridge address. The address is written to runtime/endpoint.
#   * Per-GPU ceilings are 12 CPUs and 20 GiB, and 512 MiB of --mem is reserved
#     for the host, so the container asks for less than the job holds.
#SBATCH --account=twiddle
#SBATCH --partition=gpu1
#SBATCH --job-name=atlas-flow-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/rolf/atlas-flow-demo/logs/demo-%j.out
#SBATCH --error=/data/rolf/atlas-flow-demo/logs/demo-%j.err
set -Eeuo pipefail
umask 022

readonly ROOT=${ATLAS_DEMO_ROOT:-/data/rolf/atlas-flow-demo}
readonly DATA=${ATLAS_DEMO_DATA:-/data/atlas-flow-pad-v1}
readonly IMAGE=${ATLAS_DEMO_IMAGE:-midibrave:atlas-flow-kraken-v100-v1}
readonly PORT=${ATLAS_DEMO_PORT:-18795}
readonly CHECKPOINT=$DATA/runs/pad-v1/joint/checkpoints/step-000030000.pt
readonly CONFIG_HOST=$ROOT/repo/configs/atlas_flow/kraken_pad_v1.yaml
# The container sees the config through its mount, not at the host path.
readonly CONFIG=/opt/midibrave/configs/atlas_flow/kraken_pad_v1.yaml
readonly NAME=atlas-flow-demo

[[ -s "$CHECKPOINT" ]] || { echo "checkpoint missing: $CHECKPOINT" >&2; exit 72; }
[[ -s "$CONFIG_HOST" ]] || { echo "config missing: $CONFIG_HOST" >&2; exit 72; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "run under sbatch" >&2; exit 70; }

mkdir -p "$ROOT/cache/takes" "$ROOT/cache/audition" "$ROOT/logs" "$ROOT/runtime"

# Publish the container's bridge address so the tunnel does not have to guess it.
announce() {
  local container address
  for _ in $(seq 60); do
    sleep 2
    container=$(docker ps --filter "label=kraken.name=$NAME" --filter "label=kraken.job=$SLURM_JOB_ID" \
      --format '{{.Names}}' | head -1)
    [[ -n "$container" ]] || continue
    address=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container")
    [[ -n "$address" ]] || continue
    printf '{"job":"%s","container":"%s","address":"%s","port":%s}\n' \
      "$SLURM_JOB_ID" "$container" "$address" "$PORT" > "$ROOT/runtime/endpoint.json"
    echo "endpoint http://$address:$PORT (container $container)"
    return 0
  done
  echo "could not determine the container address" >&2
}
announce &

# No --gpus: that flag takes explicit device IDs and must be a subset of the
# allocation, so omitting it inherits exactly the GPU Slurm handed us.
exec kraken-gpu-run \
  --name "$NAME" \
  --memory 19G \
  --shm-size 2g \
  --env "ATLAS_FLOW_PLAN_INTERVAL=${ATLAS_FLOW_PLAN_INTERVAL:-0.45}" \
  --mount "$DATA:$DATA:ro" \
  --mount "$ROOT/cache:$ROOT/cache" \
  --mount "$ROOT/repo/configs:/opt/midibrave/configs:ro" \
  --mount "$ROOT/repo/src:/opt/midibrave/src:ro" \
  --mount "$ROOT/repo/atlas-flow-web-demo:/opt/midibrave/atlas-flow-web-demo:ro" \
  -- "$IMAGE" python -m midibrave.atlas_flow_demo_server \
       --config "$CONFIG" \
       --checkpoint "$CHECKPOINT" \
       --cache-root "$ROOT/cache" \
       --evaluation-root "$DATA/evaluation/pad-v1" \
       --web-root /opt/midibrave/atlas-flow-web-demo \
       --device cuda:0 \
       --host 0.0.0.0 \
       --port "$PORT"
