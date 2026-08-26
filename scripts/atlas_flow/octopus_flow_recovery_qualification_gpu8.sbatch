#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --job-name=flow-recovery-qual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=110G
#SBATCH --time=00:58:00
#SBATCH --exclusive
#SBATCH --output=/data/atlas-flow-pad-v1/logs/flow-recovery-qualification-%j.out

set -euo pipefail
umask 027

readonly IMAGE=midibrave:atlas-flow-v3
readonly CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
readonly ROOT=/data/atlas-flow-pad-v1
readonly CONTRACT=$ROOT/contracts/octopus-v100
readonly PAUSE_MARKER=$CONTRACT/post-stage1-pause-requested
readonly SOURCE_OVERRIDE=/data/projects/latent-cosmos-synth/atlas-flow-recovery-20260825/MidiBrave-v2/src/midibrave/atlas_flow_train.py
readonly CHECKPOINT=$ROOT/runs/pad-v1/flow/checkpoints/step-000015000.pt
readonly EXPECTED_CHECKPOINT_SHA=3a21769e00bf3635a57c621e1d6940c00af99eb7b1c291da640b4c2b6ef9d8a6
readonly EXPECTED_SOURCE_SHA=3aaba998749b7c32c2ef025fa152250ef7e7d79016726758cbff5e6192fb0362
readonly QUALIFICATION_ROOT=$ROOT/runs/pad-v1/qualification/numeric-recovery-20260825
readonly SELECTED_PROFILE=$CONTRACT/flow-recovery-profile.txt
readonly SELECTED_BATCH=$CONTRACT/selected-flow-batch.txt
readonly SUMMARY=$CONTRACT/flow-recovery-qualification.json
readonly PROFILES=(safe_amp fp32_rescue fp32_low_lr)

mkdir -p "$QUALIFICATION_ROOT" "$CONTRACT" "$ROOT/logs"
[[ -e "$PAUSE_MARKER" ]] || { echo 'formal pause marker must remain present during isolated qualification' >&2; exit 70; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'CUDA_VISIBLE_DEVICES is empty' >&2; exit 70; }
gpu_count=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
[[ "$gpu_count" -eq 8 ]] || { echo "expected 8 allocated GPUs, got $CUDA_VISIBLE_DEVICES" >&2; exit 70; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || { echo 'qualification checkpoint hash mismatch' >&2; exit 71; }
[[ "$(sha256sum "$SOURCE_OVERRIDE" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || { echo 'source hash mismatch' >&2; exit 71; }
curl -fsS http://127.0.0.1:6006/data/logdir >/dev/null || { echo 'TensorBoard preflight failed' >&2; exit 74; }

active_container=''
cleanup_container() {
  if [[ -n "$active_container" ]]; then
    docker rm -f -v "$active_container" >/dev/null 2>&1 || true
    active_container=''
  fi
}
trap cleanup_container EXIT INT TERM

run_torch8() {
  active_container="flow-recovery-qual-${SLURM_JOB_ID}-${RANDOM}"
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
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_P2P_LEVEL=NVL \
    --env NCCL_IB_DISABLE=1 \
    --env NCCL_TIMEOUT=1800 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env MIDIBRAVE_GRAD_SYNC=per_parameter \
    --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
    --mount type=bind,src="$SOURCE_OVERRIDE",dst=/opt/midibrave/src/midibrave/atlas_flow_train.py,readonly \
    "$IMAGE" torchrun --standalone --nproc_per_node=8 "$@" || status=$?
  cleanup_container
  return "$status"
}

report_error() {
  local report=$1
  python3 - "$report" <<'PY'
import json
import sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("error") or "")
except Exception:
    print("report_unreadable")
PY
}

preferred_batch=96
[[ -s "$SELECTED_BATCH" ]] && preferred_batch=$(tr -d '[:space:]' < "$SELECTED_BATCH")
for profile in "${PROFILES[@]}"; do
  batch=$preferred_batch
  while true; do
    run_root="$QUALIFICATION_ROOT/${profile}-batch${batch}-${SLURM_JOB_ID}"
    report_first="$run_root/report-15000-18000.json"
    report_second="$run_root/report-18000-21000.json"
    checkpoint_first="$run_root/checkpoints/step-000018000.pt"
    checkpoint_second="$run_root/checkpoints/step-000021000.pt"
    mkdir -p "$run_root"
    echo "qualifying profile=$profile batch=$batch from update 15000 through 18000"
    set +e
    run_torch8 -m midibrave.atlas_flow_train qualify \
      --config "$CONFIG" --stage flow --checkpoint "$CHECKPOINT" \
      --batch-per-gpu "$batch" --grad-accum 1 \
      --warmup-updates 0 --updates 3000 --stability-gate \
      --recovery-profile "$profile" --report "$report_first" \
      --save-checkpoint "$checkpoint_first" \
      --tensorboard-log-dir "$run_root/tensorboard"
    status_first=$?
    set -e
    status=$status_first
    failed_report=$report_first
    if [[ "$status_first" -eq 0 ]]; then
      echo "qualifying exact resume profile=$profile batch=$batch from update 18000 through 21000"
      set +e
      run_torch8 -m midibrave.atlas_flow_train qualify \
        --config "$CONFIG" --stage flow --checkpoint "$checkpoint_first" \
        --batch-per-gpu "$batch" --grad-accum 1 \
        --warmup-updates 0 --updates 3000 --stability-gate \
        --recovery-profile "$profile" --report "$report_second" \
        --save-checkpoint "$checkpoint_second" \
        --tensorboard-log-dir "$run_root/tensorboard"
      status=$?
      set -e
      failed_report=$report_second
    fi
    if [[ "$status" -eq 0 ]]; then
      printf '%s\n' "$profile" > "${SELECTED_PROFILE}.tmp"
      mv "${SELECTED_PROFILE}.tmp" "$SELECTED_PROFILE"
      printf '%s\n' "$batch" > "${SELECTED_BATCH}.tmp"
      mv "${SELECTED_BATCH}.tmp" "$SELECTED_BATCH"
      python3 - "$SUMMARY" "$profile" "$batch" "$report_first" "$report_second" "$checkpoint_second" "$EXPECTED_SOURCE_SHA" "$EXPECTED_CHECKPOINT_SHA" <<'PY'
import json
import os
import sys
destination, profile, batch, report_first, report_second, checkpoint, source_sha, checkpoint_sha = sys.argv[1:]
value = {
    "schema": "midibrave.atlas-flow.recovery-qualification.v1",
    "passed": True,
    "selected_profile": profile,
    "selected_batch_per_gpu": int(batch),
    "reports": [report_first, report_second],
    "qualified_checkpoint": checkpoint,
    "source_sha256": source_sha,
    "checkpoint_sha256": checkpoint_sha,
}
temporary = destination + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, destination)
PY
      echo "qualification passed with profile=$profile batch=$batch"
      exit 0
    fi
    error=$(report_error "$failed_report")
    echo "qualification rejected profile=$profile batch=$batch status=$status error=$error" >&2
    if [[ "$batch" -ne 72 && "$error" == *"out of memory"* ]]; then
      batch=72
      continue
    fi
    break
  done
done

python3 - "$SUMMARY" "$EXPECTED_SOURCE_SHA" "$EXPECTED_CHECKPOINT_SHA" <<'PY'
import json
import os
import sys
destination, source_sha, checkpoint_sha = sys.argv[1:]
value = {
    "schema": "midibrave.atlas-flow.recovery-qualification.v1",
    "passed": False,
    "selected_profile": None,
    "source_sha256": source_sha,
    "checkpoint_sha256": checkpoint_sha,
    "error": "all recovery profiles failed isolated qualification",
}
temporary = destination + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, destination)
PY
exit 75
