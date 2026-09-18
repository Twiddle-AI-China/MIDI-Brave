#!/usr/bin/env bash
# Atlas Flow web demo on DGX Spark.
#
# Submit through the GPU queue so GPU-GUARD can trace the container:
#   SBATCH_SIGNAL=B:TERM@180 qgpu -n rolf_atlas_flow_demo -c 8 -m 32G -t 7-00:00:00 -- \
#     bash /home/rolf/projects/atlas-flow-demo/repo/scripts/atlas_flow/spark_web_demo.sh
#
# SBATCH_SIGNAL matters: without it the batch shell is signalled only at the wall
# limit itself, and a container that outlives its allocation keeps holding the GPU
# outside SLURM. Three independent guards keep that from happening — the early
# signal, a self-imposed deadline below, and a sweep of stale containers at start.
#
# The server binds loopback only; reach it with
#   ssh -N -L 18795:127.0.0.1:18795 spark
set -Eeuo pipefail
umask 022

readonly ROOT=${ATLAS_DEMO_ROOT:-/home/rolf/projects/atlas-flow-demo}
readonly IMAGE=${ATLAS_DEMO_IMAGE:-midibrave:atlas-flow-spark-gb10-v3}
readonly PORT=${ATLAS_DEMO_PORT:-18795}
readonly CHECKPOINT=$ROOT/assets/step-000030000.pt
readonly CONFIG=$ROOT/configs/demo_pad_v1.yaml
readonly STOP_MARGIN=120

[[ -s "$CHECKPOINT" ]] || { echo "checkpoint missing: $CHECKPOINT" >&2; exit 72; }
[[ -s "$CONFIG" ]] || { echo "config missing: $CONFIG" >&2; exit 72; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "no GPU allocated by SLURM" >&2; exit 70; }
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "expected exactly one GPU" >&2; exit 70; }

mkdir -p "$ROOT/cache/takes" "$ROOT/cache/audition"
container="atlas-flow-web-demo-${SLURM_JOB_ID:-manual}"

# A cancelled or timed-out job can leave its container behind, which then holds
# both the GPU and the port.
for stale in $(docker ps -aq --filter 'name=^atlas-flow-web-demo-'); do
  docker rm -f -v "$stale" >/dev/null 2>&1 || true
done

cleanup() {
  docker stop --time 15 "$container" >/dev/null 2>&1 || true
  docker rm -f -v "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Stop on our own a couple of minutes before the allocation ends, so the
# container never outlives the job even if no signal arrives.
lifetime() {
  local end stamp
  end=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
  [[ -n "$end" && "$end" != Unknown ]] || return 1
  stamp=$(date -d "$end" +%s 2>/dev/null) || return 1
  echo $(( stamp - $(date +%s) - STOP_MARGIN ))
}

run=(
  docker run --rm
  --name "$container"
  --network host
  --ipc private
  --shm-size 2g
  --cpus 8
  --memory 24g
  --memory-swap 24g
  --pids-limit 2048
  --security-opt no-new-privileges:true
  --cap-drop ALL
  --tmpfs /tmp:rw,noexec,nosuid,size=512m
  --user "$(id -u):$(id -g)"
  --gpus "device=${CUDA_VISIBLE_DEVICES}"
  --mount "type=bind,src=$ROOT/repo/src,dst=/opt/midibrave/src,readonly"
  --mount "type=bind,src=$ROOT/repo/atlas-flow-web-demo,dst=/opt/midibrave/atlas-flow-web-demo,readonly"
  --mount "type=bind,src=$ROOT/assets,dst=$ROOT/assets,readonly"
  --mount "type=bind,src=$ROOT/configs,dst=$ROOT/configs,readonly"
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache"
  "$IMAGE" python -m midibrave.atlas_flow_demo_server
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --cache-root "$ROOT/cache"
  --evaluation-root "$ROOT/assets/evaluation"
  --web-root /opt/midibrave/atlas-flow-web-demo
  --device cuda:0
  --host 127.0.0.1
  --port "$PORT"
)

ttl=$(lifetime || true)
if [[ -n "${ttl:-}" && "$ttl" -gt 60 ]]; then
  echo "runtime deadline: ${ttl}s (stops ${STOP_MARGIN}s before the allocation ends)"
  timeout --signal=TERM "${ttl}s" "${run[@]}" &
else
  "${run[@]}" &
fi
# Not exec, and backgrounded: the trap above must stay able to stop the container.
runner=$!
wait "$runner"
