#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=atlas-flow-qual-v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/atlas-flow-pad-v1/logs/spark-live-qualification-v5-%j.out

set -Eeuo pipefail
umask 027

readonly IMAGE=${ATLAS_FLOW_IMAGE:-midibrave:atlas-flow-spark-gb10-v3}
readonly ROOT=${ATLAS_FLOW_ROOT:-/data/atlas-flow-pad-v1}
readonly CONFIG=/opt/midibrave/configs/atlas_flow/spark_pad_v1.yaml
readonly CHECKPOINT=$ROOT/runs/pad-v1/joint/checkpoints/step-000030000.pt
readonly CONTRACT=$ROOT/contracts/spark-gb10/v5
readonly REPORT=$CONTRACT/qualification-${SLURM_JOB_ID}.json
readonly MARKER=$CONTRACT/qualification.complete

mkdir -p "$ROOT/logs" "$CONTRACT"
[[ -s "$CHECKPOINT" ]] || { echo "joint checkpoint is missing" >&2; exit 72; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "SLURM did not allocate a GPU" >&2; exit 70; }
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "expected exactly one GPU" >&2; exit 70; }

container="atlas-flow-qual-v5-${SLURM_JOB_ID}"
cleanup() {
  docker rm -f -v "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker run --rm \
  --name "$container" \
  --network none \
  --ipc private \
  --shm-size 3g \
  --cpus 8 \
  --memory 16g \
  --memory-swap 16g \
  --pids-limit 2048 \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --user "$(id -u):$(id -g)" \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --mount type=bind,src="$ROOT",dst="$ROOT",bind-propagation=rprivate \
  "$IMAGE" python /opt/midibrave/scripts/atlas_flow/spark_runtime_qualification.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --report "$REPORT" \
    --device cuda:0

checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
printf '{"state":"passed","job_id":"%s","checkpoint_sha256":"%s","image":"%s","image_id":"%s","report":"%s"}\n' \
  "$SLURM_JOB_ID" "$checkpoint_sha" "$IMAGE" "$image_id" "$(basename "$REPORT")" \
  > "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
