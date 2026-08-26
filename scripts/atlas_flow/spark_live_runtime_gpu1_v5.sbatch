#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=atlas-flow-live-v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=30-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/atlas-flow-pad-v1/logs/spark-live-runtime-v5-%j.out

set -Eeuo pipefail
umask 027

readonly IMAGE=${ATLAS_FLOW_IMAGE:-midibrave:atlas-flow-spark-gb10-v3}
readonly ROOT=${ATLAS_FLOW_ROOT:-/data/atlas-flow-pad-v1}
readonly CONFIG=/opt/midibrave/configs/atlas_flow/spark_pad_v1.yaml
readonly CHECKPOINT=$ROOT/runs/pad-v1/joint/checkpoints/step-000030000.pt
readonly QUALIFICATION=$ROOT/contracts/spark-gb10/v5/qualification.complete
readonly RUNTIME_PORT=${ATLAS_FLOW_RUNTIME_PORT:-8791}

mkdir -p "$ROOT/logs"
[[ -s "$CHECKPOINT" ]] || { echo "joint checkpoint is missing" >&2; exit 72; }
[[ -s "$QUALIFICATION" ]] || { echo "Spark runtime qualification is missing" >&2; exit 73; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "SLURM did not allocate a GPU" >&2; exit 70; }
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "expected exactly one GPU" >&2; exit 70; }
[[ "$RUNTIME_PORT" =~ ^[0-9]+$ ]] || { echo "runtime port must be numeric" >&2; exit 64; }

container="atlas-flow-live-v5-${SLURM_JOB_ID}"
cleanup() {
  docker stop --time 20 "$container" >/dev/null 2>&1 || true
  docker rm -f -v "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker run --rm \
  --name "$container" \
  --network host \
  --ipc private \
  --shm-size 3g \
  --cpus 8 \
  --memory 16g \
  --memory-swap 16g \
  --pids-limit 2048 \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --user "$(id -u):$(id -g)" \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --mount type=bind,src="$ROOT",dst="$ROOT",readonly,bind-propagation=rprivate \
  "$IMAGE" python -m midibrave.atlas_flow_runtime_server \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port "$RUNTIME_PORT"
