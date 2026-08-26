#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --job-name=atlas-flow-pad-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/data/atlas-flow-pad-v1/logs/gpu8-%j.out

set -euo pipefail
umask 027

IMAGE=midibrave:atlas-flow-v3
CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
ROOT=/data/atlas-flow-pad-v1
RUN_ROOT=$ROOT/runs/pad-v1
EVALUATION=$ROOT/evaluation/pad-v1
FLOW_BATCH_FILE=$ROOT/contracts/selected-flow-batch.txt
STAGE1_BATCH_FILE=$ROOT/contracts/selected-stage1-batch.txt
JOINT_BATCH_FILE=$ROOT/contracts/selected-joint-batch.txt
mkdir -p "$ROOT/logs" "$ROOT/contracts" "$EVALUATION"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CUDA_VISIBLE_DEVICES is empty" >&2; exit 70; }
gpu_count=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
[[ "$gpu_count" -eq 8 ]] || { echo "expected exactly 8 allocated GPUs, got $CUDA_VISIBLE_DEVICES" >&2; exit 70; }

requeue_requested=0
active_container=""
cleanup_container() {
  if [[ -n "$active_container" ]]; then
    docker rm -f -v "$active_container" >/dev/null 2>&1 || true
    active_container=""
  fi
}
on_terminate() {
  cleanup_container
  exit 143
}
on_requeue() {
  if [[ "$requeue_requested" -eq 0 ]]; then
    requeue_requested=1
    echo "checkpoint window requested; requeueing job $SLURM_JOB_ID" >&2
    cleanup_container
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
}
trap on_requeue USR1
trap cleanup_container EXIT
trap on_terminate INT TERM

run_gpu() {
  active_container="atlas-flow-${SLURM_JOB_ID}-${RANDOM}"
  local status=0
  docker run --rm \
    --name "$active_container" \
    --network none \
    --ipc private \
    --shm-size 16g \
    --cpus 92 \
    --memory 106g \
    --memory-swap 106g \
    --pids-limit 16384 \
    --user "$(id -u):$(id -g)" \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
    "$IMAGE" "$@" || status=$?
  cleanup_container
  return "$status"
}

run_torch8() {
  run_gpu torchrun --standalone --nproc_per_node=8 "$@"
}

checkpoint_arg() {
  local latest=$1
  local initialize=${2:-}
  if [[ -s "$latest" ]]; then
    printf '%s\n' --resume "$latest"
  elif [[ -n "$initialize" ]]; then
    printf '%s\n' --initialize "$initialize"
  fi
}

run_train() {
  local stage=$1
  local maximum=$2
  local batch=${3:-}
  local initialize=${4:-}
  local latest="$RUN_ROOT/$stage/checkpoints/latest.pt"
  mapfile -t restore < <(checkpoint_arg "$latest" "$initialize")
  local command=(-m midibrave.atlas_flow_train train --config "$CONFIG" --stage "$stage" --max-updates "$maximum")
  if [[ -n "$batch" ]]; then command+=(--batch-per-gpu "$batch"); fi
  command+=("${restore[@]}")
  run_torch8 "${command[@]}"
}

run_gpu python -c "from midibrave.atlas_flow_config import load_atlas_flow_config; from midibrave.atlas_flow_data import manifest_audit; c=load_atlas_flow_config('$CONFIG'); print(manifest_audit(c.data, check_audio=True))"

run_torch8 -m midibrave.atlas_flow_train benchmark \
  --kind audio --config "$CONFIG" --batches 32,24,16,8,4 --updates 3 \
  --report "$ROOT/contracts/stage1-batch-sweep.json"
run_gpu python /opt/midibrave/scripts/atlas_flow/select_flow_batch.py \
  --report "$ROOT/contracts/stage1-batch-sweep.json" --minimum 4 \
  --output "$STAGE1_BATCH_FILE"
stage1_batch=$(tr -d '[:space:]' < "$STAGE1_BATCH_FILE")

run_torch8 -m midibrave.atlas_flow_train benchmark \
  --config "$CONFIG" --batches 96,88,80,72 --updates 5 \
  --report "$ROOT/contracts/flow-batch-sweep.json"
run_gpu python /opt/midibrave/scripts/atlas_flow/select_flow_batch.py \
  --report "$ROOT/contracts/flow-batch-sweep.json" --output "$FLOW_BATCH_FILE"
flow_batch=$(tr -d '[:space:]' < "$FLOW_BATCH_FILE")

# Stage 1 begins from random initialization. The first 200 updates are an
# overfit/numerics qualification; the same checkpoint then continues to 60k.
run_train stage1 200 "$stage1_batch"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/stage1/metrics.jsonl" --minimum-update 200 \
  --minimum-relative-decrease 0.02 --report "$ROOT/contracts/stage1-overfit-gate.json"
run_train stage1 60000 "$stage1_batch"

run_torch8 -m midibrave.atlas_flow_train cache-trajectories \
  --config "$CONFIG" --checkpoint "$RUN_ROOT/stage1/checkpoints/latest.pt" --batch-size 32

# Flow qualification and formal training share one checkpoint lineage. If a
# selected larger batch OOMs in real data, retry at the historical safe 72.
set +e
run_train flow 5000 "$flow_batch" "$RUN_ROOT/stage1/checkpoints/latest.pt"
flow_rc=$?
set -e
if [[ "$flow_rc" -ne 0 && "$flow_batch" -ne 72 ]]; then
  echo "selected flow batch $flow_batch failed; retrying approved batch 72" >&2
  flow_batch=72
  printf '72\n' > "$FLOW_BATCH_FILE"
  run_train flow 5000 72 "$RUN_ROOT/stage1/checkpoints/latest.pt"
elif [[ "$flow_rc" -ne 0 ]]; then
  exit "$flow_rc"
fi
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/flow/metrics.jsonl" --minimum-update 5000 \
  --minimum-relative-decrease 0.01 --report "$ROOT/contracts/flow-qualification-gate.json"
run_train flow 100000 "$flow_batch" "$RUN_ROOT/stage1/checkpoints/latest.pt"

# Joint consumes both loaders. In the trainer a shared batch B becomes an
# audio batch B/2 and a flow batch B/4, so reusing flow_batch=96 would create
# audio batch 48 and exceed a 16 GiB V100. Qualify the largest real joint batch
# in an isolated container and retain an explicit, restart-safe selection.
if [[ -s "$RUN_ROOT/joint/checkpoints/latest.pt" && -s "$JOINT_BATCH_FILE" ]]; then
  joint_batch=$(tr -d '[:space:]' < "$JOINT_BATCH_FILE")
else
  joint_batch=""
  for candidate in 48 40 32 24; do
    echo "qualifying joint batch $candidate (audio=$((candidate / 2)), flow=$((candidate / 4)))"
    set +e
    run_train joint 100 "$candidate" "$RUN_ROOT/flow/checkpoints/latest.pt"
    joint_rc=$?
    set -e
    if [[ "$joint_rc" -eq 0 ]]; then
      joint_batch=$candidate
      printf '%s\n' "$joint_batch" > "$JOINT_BATCH_FILE"
      break
    fi
    echo "joint batch $candidate failed with status $joint_rc; trying a smaller batch" >&2
  done
  [[ -n "$joint_batch" ]] || { echo "no joint batch candidate passed" >&2; exit 71; }
  run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
    --metrics "$RUN_ROOT/joint/metrics.jsonl" --minimum-update 100 \
    --require-all-modules --report "$ROOT/contracts/joint-qualification-gate.json"
fi
run_train joint 30000 "$joint_batch" "$RUN_ROOT/flow/checkpoints/latest.pt"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/joint/metrics.jsonl" --minimum-update 30000 \
  --require-all-modules --report "$ROOT/contracts/joint-all-modules-gate.json"

run_gpu python -m midibrave.atlas_flow_evaluate \
  --config "$CONFIG" --checkpoint "$RUN_ROOT/joint/checkpoints/latest.pt" \
  --output "$EVALUATION" --device cuda:0
run_gpu python -m midibrave.atlas_flow_dashboard \
  --evaluation-root "$EVALUATION" \
  --atlas /data/atlas-flow-pad-v1/atlas/pad-top50-atlas.npz \
  --manifest /data/atlas-flow-pad-v1/manifests/pad-top50-training.jsonl

touch "$ROOT/contracts/gpu-pipeline.complete"
