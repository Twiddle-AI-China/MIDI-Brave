#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=atlas-flow-pad-spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=110000M
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/data/atlas-flow-pad-v1/logs/spark-%j.out

set -euo pipefail
umask 027

readonly IMAGE=midibrave:atlas-flow-spark-gb10-v2
readonly CONFIG=/opt/midibrave/configs/atlas_flow/spark_pad_v1.yaml
readonly ROOT=/data/atlas-flow-pad-v1
readonly RUN_ROOT=$ROOT/runs/pad-v1
readonly EVALUATION=$ROOT/evaluation/pad-v1
readonly EXPECTED_STAGE1_SHA256=5058043a3665e6e34643764705224a8dacb9ef8c318837382b315b81dd94a404
readonly MIGRATED_STAGE1=$RUN_ROOT/stage1/checkpoints/step-000050000.pt
readonly PAUSE_MARKER=$ROOT/contracts/spark-gb10/pause-requested
readonly CONTRACT_ROOT=$ROOT/contracts/spark-gb10/v2
mkdir -p "$ROOT/logs" "$CONTRACT_ROOT" "$EVALUATION"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CUDA_VISIBLE_DEVICES is empty" >&2; exit 70; }
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "expected exactly one allocated GPU" >&2; exit 70; }

requeue_requested=0
active_container=""
cleanup_container() {
  if [[ -n "$active_container" ]]; then
    docker rm -f -v "$active_container" >/dev/null 2>&1 || true
    active_container=""
  fi
}
on_usr1() {
  requeue_requested=1
  echo "graceful checkpoint requested for SLURM job $SLURM_JOB_ID" >&2
  if [[ -n "$active_container" ]]; then
    docker kill --signal=USR1 "$active_container" >/dev/null 2>&1 || true
  fi
}
on_terminate() {
  cleanup_container
  exit 143
}
trap on_usr1 USR1
trap cleanup_container EXIT
trap on_terminate INT TERM

run_gpu() {
  active_container="atlas-flow-spark-${SLURM_JOB_ID}-${RANDOM}"
  local status=0
  docker run --rm \
    --name "$active_container" \
    --network none \
    --ipc private \
    --shm-size 16g \
    --cpus 18 \
    --memory 106g \
    --memory-swap 106g \
    --pids-limit 16384 \
    --user "$(id -u):$(id -g)" \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
    "$IMAGE" "$@" || status=$?
  active_container=""
  if [[ -e "$PAUSE_MARKER" ]]; then
    echo "manual pause completed; submit the job again with spark_resume_job.sh" >&2
    exit 0
  fi
  if [[ "$requeue_requested" -eq 1 ]]; then
    echo "requeueing SLURM job $SLURM_JOB_ID after checkpoint handoff" >&2
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  return "$status"
}

read_contract() {
  local stage=$1
  local selected="$CONTRACT_ROOT/${stage}-batch.txt"
  [[ -s "$selected" ]] || return 1
  read -r SELECTED_MICRO SELECTED_ACCUM < "$selected"
  [[ "$SELECTED_MICRO" =~ ^[0-9]+$ && "$SELECTED_ACCUM" =~ ^[0-9]+$ ]]
}

qualify_contract() {
  local stage=$1
  local checkpoint=$2
  local baseline=$3
  shift 3
  local selected="$CONTRACT_ROOT/${stage}-batch.txt"
  local contract="$CONTRACT_ROOT/${stage}-contract.json"
  local ranked="$CONTRACT_ROOT/${stage}-ranked.txt"
  if read_contract "$stage"; then
    [[ -s "$contract" && -s "$ranked" ]] || { echo "$stage v2 contract is incomplete" >&2; exit 72; }
    echo "reusing throughput-qualified $stage contract: micro=$SELECTED_MICRO accum=$SELECTED_ACCUM"
    return
  fi
  local spec micro accum report status checkpoint_sha
  local reports=()
  for spec in "$@"; do
    IFS=: read -r micro accum <<< "$spec"
    report="$CONTRACT_ROOT/${stage}-qualify-${micro}x${accum}-${SLURM_JOB_ID}.json"
    echo "throughput-qualifying $stage on real data: micro=$micro accum=$accum"
    set +e
    run_gpu python -m midibrave.atlas_flow_train qualify \
      --config "$CONFIG" --stage "$stage" --checkpoint "$checkpoint" \
      --batch-per-gpu "$micro" --grad-accum "$accum" \
      --warmup-updates 2 --updates 5 --report "$report"
    status=$?
    set -e
    if [[ -s "$report" ]]; then
      reports+=("$report")
    fi
    if [[ "$status" -ne 0 ]]; then
      echo "$stage qualification $micro x $accum rejected with status $status" >&2
    fi
  done
  [[ "${#reports[@]}" -gt 0 ]] || { echo "no $stage candidate produced a valid report" >&2; exit 72; }
  checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
  local selector=(
    python /opt/midibrave/scripts/atlas_flow/select_spark_batch_contract.py
    --stage "$stage" --baseline "$baseline"
    --minimum-headroom-mib 24576 --minimum-system-available-mib 20480
    --minimum-relative-gain 0.05 --tie-fraction 0.03
    --checkpoint-sha256 "$checkpoint_sha" --image "$IMAGE"
    --output "$contract" --text-output "$selected" --ranked-output "$ranked"
  )
  for report in "${reports[@]}"; do selector+=(--report "$report"); done
  run_gpu "${selector[@]}"
  read_contract "$stage" || { echo "$stage selector did not write a valid contract" >&2; exit 72; }
  echo "selected $stage contract: micro=$SELECTED_MICRO accum=$SELECTED_ACCUM"
}

run_train() {
  local stage=$1
  local maximum=$2
  local micro=$3
  local accum=$4
  local initialize=${5:-}
  local latest="$RUN_ROOT/$stage/checkpoints/latest.pt"
  local restore=()
  if [[ -s "$latest" ]]; then
    restore=(--resume "$latest")
  elif [[ -n "$initialize" ]]; then
    restore=(--initialize "$initialize")
  fi
  run_gpu python -m midibrave.atlas_flow_train train \
    --config "$CONFIG" --stage "$stage" --max-updates "$maximum" \
    --batch-per-gpu "$micro" --grad-accum "$accum" \
    --session-id "spark-${SLURM_JOB_ID}-${stage}" "${restore[@]}"
}

run_train_with_fallback() {
  local stage=$1
  local maximum=$2
  local initialize=${3:-}
  local ranked="$CONTRACT_ROOT/${stage}-ranked.txt"
  local micro accum status
  [[ -s "$ranked" ]] || { echo "missing ranked candidates for $stage" >&2; exit 72; }
  while read -r micro accum; do
    [[ -n "$micro" && -n "$accum" ]] || continue
    echo "training $stage with micro=$micro accum=$accum"
    set +e
    run_train "$stage" "$maximum" "$micro" "$accum" "$initialize"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      printf '%s %s\n' "$micro" "$accum" > "$CONTRACT_ROOT/${stage}-active-batch.txt"
      return
    fi
    if [[ "$status" -ne 72 && "$status" -ne 137 ]]; then
      echo "$stage failed with non-OOM status $status; refusing to mask it" >&2
      return "$status"
    fi
    echo "$stage candidate micro=$micro accum=$accum OOMed; trying next qualified contract" >&2
  done < "$ranked"
  echo "all throughput-qualified $stage contracts failed during formal training" >&2
  exit 72
}

readonly STAGE1_LATEST="$RUN_ROOT/stage1/checkpoints/latest.pt"
[[ -s "$STAGE1_LATEST" ]] || { echo "migrated Stage1 checkpoint is missing" >&2; exit 73; }
[[ -s "$MIGRATED_STAGE1" ]] || { echo "immutable Stage1 step 50000 is missing" >&2; exit 73; }
printf '%s  %s\n' "$EXPECTED_STAGE1_SHA256" "$MIGRATED_STAGE1" | sha256sum --check --status

run_gpu python -c "from midibrave.atlas_flow_config import load_atlas_flow_config; from midibrave.atlas_flow_data import manifest_audit; c=load_atlas_flow_config('$CONFIG'); print(manifest_audit(c.data, check_audio=True))"

# Every Stage1 candidate preserves global batch 256.
qualify_contract stage1 "$STAGE1_LATEST" 32:8 256:1 128:2 64:4 32:8
run_train_with_fallback stage1 60000

run_gpu python -m midibrave.atlas_flow_train cache-trajectories \
  --config "$CONFIG" --checkpoint "$STAGE1_LATEST" --batch-size 64

# Every Flow candidate preserves global batch 768.
qualify_contract flow "$STAGE1_LATEST" 96:8 768:1 384:2 192:4 96:8
run_train_with_fallback flow 5000 "$STAGE1_LATEST"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/flow/metrics.jsonl" --minimum-update 5000 \
  --minimum-relative-decrease 0.01 \
  --report "$ROOT/contracts/spark-gb10/flow-5000-gate.json"
run_train_with_fallback flow 100000 "$STAGE1_LATEST"

readonly FLOW_LATEST="$RUN_ROOT/flow/checkpoints/latest.pt"
# Every Joint candidate preserves audio global 192 and Flow global 96.
qualify_contract joint "$FLOW_LATEST" 48:8 384:1 192:2 96:4 48:8
run_train_with_fallback joint 30000 "$FLOW_LATEST"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/joint/metrics.jsonl" --minimum-update 30000 \
  --require-all-modules \
  --report "$ROOT/contracts/spark-gb10/joint-all-modules-gate.json"

run_gpu python -m midibrave.atlas_flow_evaluate \
  --config "$CONFIG" --checkpoint "$RUN_ROOT/joint/checkpoints/latest.pt" \
  --output "$EVALUATION" --device cuda:0
run_gpu python -m midibrave.atlas_flow_dashboard \
  --evaluation-root "$EVALUATION" \
  --atlas /data/atlas-flow-pad-v1/atlas/pad-top50-atlas.npz \
  --manifest /data/atlas-flow-pad-v1/manifests/pad-top50-training.jsonl

touch "$ROOT/contracts/spark-gb10/gpu-pipeline.complete"
