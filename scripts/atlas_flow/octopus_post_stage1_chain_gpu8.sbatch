#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --job-name=atlas-post-s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/atlas-flow-pad-v1/logs/post-stage1-chain-%j.out

set -euo pipefail
umask 027

readonly IMAGE=midibrave:atlas-flow-v3
readonly CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
readonly ROOT=/data/atlas-flow-pad-v1
readonly RUN_ROOT=$ROOT/runs/pad-v1
readonly EVALUATION=$ROOT/evaluation/pad-v1
readonly CONTRACT=$ROOT/contracts/octopus-v100
readonly SOURCE_OVERRIDE=/data/projects/latent-cosmos-synth/atlas-flow-recovery-20260825/MidiBrave-v2/src/midibrave/atlas_flow_train.py
readonly STAGE1=$RUN_ROOT/stage1/checkpoints/latest.pt
readonly FLOW=$RUN_ROOT/flow/checkpoints/latest.pt
readonly JOINT=$RUN_ROOT/joint/checkpoints/latest.pt
readonly FLOW_BATCH_FILE=$CONTRACT/selected-flow-batch.txt
readonly JOINT_BATCH_FILE=$CONTRACT/selected-joint-batch.txt
readonly CACHE_MARKER=$CONTRACT/trajectory-cache-stage1-sha256.txt
readonly PAUSE_MARKER=$CONTRACT/post-stage1-pause-requested
readonly COMPLETE=$CONTRACT/post-stage1-chain.complete
readonly EXPECTED_STAGE1_SHA=4bea58be4921181e5dca5e54a2dacd7370b0b4ca821042bd07463c4d8c326bbb
readonly EXPECTED_SOURCE_SHA=c54aa079f4a3e772fa8f2240bda4e16ce1e837f4eb7a69b2f0ae1d48257a5fd1
readonly RECOVERY_PROFILES=(safe_amp fp32_rescue fp32_low_lr)

mkdir -p "$ROOT/logs" "$CONTRACT" "$EVALUATION"
rm -f "$COMPLETE"

cleanup_stale_job_containers() {
  local name owner_job
  while IFS= read -r name; do
    [[ "$name" =~ ^atlas-post-s1-([0-9]+)-[0-9]+$ ]] || continue
    owner_job=${BASH_REMATCH[1]}
    [[ "$owner_job" != "$SLURM_JOB_ID" ]] || continue
    if ! squeue -h -j "$owner_job" 2>/dev/null | grep -q .; then
      echo "removing stale post-stage1 container $name from inactive job $owner_job"
      docker rm -f -v "$name" >/dev/null
    fi
  done < <(docker ps --format '{{.Names}}')
}

cleanup_stale_job_containers
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'CUDA_VISIBLE_DEVICES is empty' >&2; exit 70; }
gpu_count=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
[[ "$gpu_count" -eq 8 ]] || { echo "expected 8 allocated GPUs, got $CUDA_VISIBLE_DEVICES" >&2; exit 70; }
stage1_sha=$(sha256sum "$STAGE1" | awk '{print $1}')
[[ "$stage1_sha" == "$EXPECTED_STAGE1_SHA" ]] || { echo "stage1 hash mismatch: $stage1_sha" >&2; exit 71; }
[[ "$(sha256sum "$SOURCE_OVERRIDE" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || { echo 'source hash mismatch' >&2; exit 71; }
[[ ! -e "$PAUSE_MARKER" ]] || { echo 'formal chain is paused; waiting for explicit operator release'; exit 0; }
curl -fsS http://127.0.0.1:6006/data/logdir >/dev/null || { echo 'TensorBoard preflight failed' >&2; exit 74; }

active_container=''
cleanup_container() {
  if [[ -n "$active_container" ]]; then
    docker rm -f -v "$active_container" >/dev/null 2>&1 || true
    active_container=''
  fi
}
forward_stop() {
  if [[ -n "$active_container" ]]; then
    local worker_count
    worker_count=$(docker exec "$active_container" sh -lc 'pgrep -P 1 | wc -l' 2>/dev/null | tr -d '[:space:]')
    if [[ "$worker_count" == 8 ]]; then
      docker exec "$active_container" sh -lc \
        'set -- $(pgrep -P 1); [ "$#" -eq 8 ] || exit 1; kill -USR1 "$@"' \
        >/dev/null 2>&1 || true
    else
      echo "cannot forward graceful stop: expected 8 training workers, found ${worker_count:-0}" >&2
    fi
  fi
}
trap cleanup_container EXIT INT TERM
trap forward_stop USR1

run_gpu() {
  active_container="atlas-post-s1-${SLURM_JOB_ID}-${RANDOM}"
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
    "$IMAGE" "$@" || status=$?
  cleanup_container
  if [[ -e "$PAUSE_MARKER" ]]; then
    echo 'post-stage1 chain paused by request; resubmit after explicit release'
    exit 0
  fi
  return "$status"
}

run_torch8() {
  run_gpu torchrun --standalone --nproc_per_node=8 "$@"
}

profile_index() {
  local selected=$1
  local index
  for index in "${!RECOVERY_PROFILES[@]}"; do
    if [[ "${RECOVERY_PROFILES[$index]}" == "$selected" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  return 1
}

state_status() {
  local state=$1
  if [[ ! -s "$state" ]]; then
    echo missing
    return 0
  fi
  python3 - "$state" <<'PY'
import json
import sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))
except Exception:
    print("invalid")
PY
}

write_launch_state() {
  local state=$1
  local stage=$2
  local profile=$3
  local batch=$4
  local temporary="${state}.tmp"
  printf '{"schema":"midibrave.atlas-flow.recovery-state.v1","status":"launching","stage":"%s","recovery_profile":"%s","batch_per_gpu":%s,"slurm_job_id":"%s"}\n' \
    "$stage" "$profile" "$batch" "$SLURM_JOB_ID" > "$temporary"
  mv "$temporary" "$state"
}

next_batch() {
  local stage=$1
  local current=$2
  local values=()
  if [[ "$stage" == flow ]]; then
    values=(96 88 80 72)
  else
    values=(48 40 32 24)
  fi
  local found=0 value
  for value in "${values[@]}"; do
    if (( found )); then
      printf '%s\n' "$value"
      return 0
    fi
    [[ "$value" == "$current" ]] && found=1
  done
  return 1
}

newest_stage_checkpoint() {
  local stage=$1
  python3 - "$RUN_ROOT/$stage/checkpoints" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
candidates = []
for pattern in ("step-*.pt", "recovery/step-*.pt"):
    for path in root.glob(pattern):
        match = re.match(r"step-(\d+)", path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), str(path)))
if not candidates:
    raise SystemExit(1)
print(max(candidates)[1])
PY
}

run_train_recovering() {
  local stage=$1
  local maximum=$2
  local batch_file=$3
  local initialize=${4:-}
  local state="$CONTRACT/${stage}-recovery-state.json"
  local profile_file="$CONTRACT/${stage}-recovery-profile.txt"
  local selected_profile=safe_amp
  [[ -s "$profile_file" ]] && selected_profile=$(tr -d '[:space:]' < "$profile_file")
  local tier
  tier=$(profile_index "$selected_profile") || { echo "invalid recovery profile: $selected_profile" >&2; exit 71; }
  local batch
  batch=$(tr -d '[:space:]' < "$batch_file")

  while true; do
    local newest_checkpoint=''
    local restore=()
    if newest_checkpoint=$(newest_stage_checkpoint "$stage"); then
      echo "$stage selecting newest checkpoint: $newest_checkpoint"
      restore=(--resume "$newest_checkpoint")
    elif [[ -n "$initialize" ]]; then
      restore=(--initialize "$initialize")
    fi

    write_launch_state "$state" "$stage" "$selected_profile" "$batch"
    set +e
    run_torch8 -m midibrave.atlas_flow_train train \
      --config "$CONFIG" --stage "$stage" --max-updates "$maximum" \
      --batch-per-gpu "$batch" --grad-accum 1 \
      --session-id "octopus-${SLURM_JOB_ID}-${stage}-${selected_profile}" \
      --recovery-profile "$selected_profile" --recovery-state "$state" \
      "${restore[@]}"
    local command_status=$?
    set -e
    local training_status
    training_status=$(state_status "$state")

    if [[ "$command_status" -eq 0 && "$training_status" == complete ]]; then
      return 0
    fi
    if [[ "$training_status" == recovery_required ]]; then
      if (( tier + 1 >= ${#RECOVERY_PROFILES[@]} )); then
        echo "$stage exhausted all numeric recovery profiles" >&2
        return 75
      fi
      tier=$((tier + 1))
      selected_profile=${RECOVERY_PROFILES[$tier]}
      printf '%s\n' "$selected_profile" > "${profile_file}.tmp"
      mv "${profile_file}.tmp" "$profile_file"
      echo "$stage rolling back to the last healthy checkpoint with profile $selected_profile"
      continue
    fi
    if [[ "$training_status" == oom ]]; then
      local smaller
      if ! smaller=$(next_batch "$stage" "$batch"); then
        echo "$stage exhausted the approved OOM batch ladder at batch $batch" >&2
        return 72
      fi
      batch=$smaller
      printf '%s\n' "$batch" > "${batch_file}.tmp"
      mv "${batch_file}.tmp" "$batch_file"
      echo "$stage retrying after OOM with batch $batch"
      continue
    fi
    if [[ "$training_status" == paused ]]; then
      return 76
    fi
    [[ "$command_status" -ne 0 ]] && return "$command_status"
    echo "$stage exited without a complete recovery state" >&2
    return 76
  done
}

checkpoint_update() {
  local checkpoint=$1
  run_gpu python -c "import torch; c=torch.load('$checkpoint',map_location='cpu',weights_only=False); print({'stage':c.get('stage'),'update':c.get('update'),'checkpoint':'$checkpoint'}); assert int(c.get('update',-1))>0"
}

echo '[1/7] rebuilding all 2700 trajectories and the atlas from stage1-60000'
if [[ -s "$CACHE_MARKER" ]] && [[ "$(tr -d '[:space:]' < "$CACHE_MARKER")" == "$stage1_sha" ]] && [[ -s "$ROOT/atlas/pad-top50-atlas.npz" ]]; then
  echo 'trajectory cache already matches current stage1; skipping rebuild'
else
  run_torch8 -m midibrave.atlas_flow_train cache-trajectories \
    --config "$CONFIG" --checkpoint "$STAGE1" --batch-size 32
  printf '%s\n' "$stage1_sha" > "$CACHE_MARKER.tmp"
  mv "$CACHE_MARKER.tmp" "$CACHE_MARKER"
fi

if [[ -s "$FLOW" ]] && [[ -s "$FLOW_BATCH_FILE" ]]; then
  flow_batch=$(tr -d '[:space:]' < "$FLOW_BATCH_FILE")
  echo "[2/7] resuming existing flow checkpoint with batch $flow_batch"
else
  echo '[2/7] qualifying flow batch candidates'
  flow_batch=''
  for candidate in 96 88 80 72; do
    report="$CONTRACT/flow-qualification-${candidate}-${SLURM_JOB_ID}.json"
    set +e
    run_torch8 -m midibrave.atlas_flow_train qualify \
      --config "$CONFIG" --stage flow --checkpoint "$STAGE1" \
      --batch-per-gpu "$candidate" --grad-accum 1 \
      --warmup-updates 1 --updates 3 --report "$report" \
      --recovery-profile safe_amp \
      --tensorboard-log-dir "$RUN_ROOT/qualification/flow-batch-${candidate}-${SLURM_JOB_ID}"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      flow_batch=$candidate
      printf '%s\n' "$flow_batch" > "$FLOW_BATCH_FILE"
      break
    fi
  done
  [[ -n "$flow_batch" ]] || { echo 'no flow batch candidate qualified' >&2; exit 72; }
fi

echo "[3/7] flow qualification gate to update 5000 with batch $flow_batch"
run_train_recovering flow 5000 "$FLOW_BATCH_FILE" "$STAGE1"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/flow/metrics.jsonl" --minimum-update 5000 \
  --minimum-relative-decrease 0.01 --report "$CONTRACT/flow-5000-gate.json"

echo "[4/7] formal flow training to update 100000 with batch $(tr -d '[:space:]' < "$FLOW_BATCH_FILE")"
run_train_recovering flow 100000 "$FLOW_BATCH_FILE" "$STAGE1"
checkpoint_update "$FLOW"
touch "$CONTRACT/flow-100000.complete"

if [[ -s "$JOINT" ]] && [[ -s "$JOINT_BATCH_FILE" ]]; then
  joint_batch=$(tr -d '[:space:]' < "$JOINT_BATCH_FILE")
else
  echo '[5/7] qualifying joint batch candidates'
  joint_batch=''
  for candidate in 48 40 32 24; do
    report="$CONTRACT/joint-qualification-${candidate}-${SLURM_JOB_ID}.json"
    set +e
    run_torch8 -m midibrave.atlas_flow_train qualify \
      --config "$CONFIG" --stage joint --checkpoint "$FLOW" \
      --batch-per-gpu "$candidate" --grad-accum 1 \
      --warmup-updates 1 --updates 3 --report "$report" \
      --recovery-profile safe_amp \
      --tensorboard-log-dir "$RUN_ROOT/qualification/joint-batch-${candidate}-${SLURM_JOB_ID}"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      joint_batch=$candidate
      printf '%s\n' "$joint_batch" > "$JOINT_BATCH_FILE"
      break
    fi
  done
  [[ -n "$joint_batch" ]] || { echo 'no joint batch candidate qualified' >&2; exit 73; }
fi

# Joint uses a 1000-update linear warmup.  Evaluate only after warmup and allow
# a 5% smoothed stochastic fluctuation at this health/connectivity gate; the
# final 30000-update gate below still requires a non-increasing loss trend.
echo "[6/7] joint qualification gate to update 1000 with batch $joint_batch"
run_train_recovering joint 1000 "$JOINT_BATCH_FILE" "$FLOW"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/joint/metrics.jsonl" --minimum-update 1000 \
  --minimum-relative-decrease -0.05 \
  --require-all-modules --report "$CONTRACT/joint-1000-gate.json"

echo "[6/7] formal joint training to update 30000 with batch $(tr -d '[:space:]' < "$JOINT_BATCH_FILE")"
run_train_recovering joint 30000 "$JOINT_BATCH_FILE" "$FLOW"
run_gpu python /opt/midibrave/scripts/atlas_flow/check_training_gate.py \
  --metrics "$RUN_ROOT/joint/metrics.jsonl" --minimum-update 30000 \
  --require-all-modules --report "$CONTRACT/joint-30000-gate.json"
checkpoint_update "$JOINT"
touch "$CONTRACT/joint-30000.complete"

echo '[7/7] evaluation and dashboard generation'
run_gpu python -m midibrave.atlas_flow_evaluate \
  --config "$CONFIG" --checkpoint "$JOINT" \
  --output "$EVALUATION" --device cuda:0
run_gpu python -m midibrave.atlas_flow_dashboard \
  --evaluation-root "$EVALUATION" \
  --atlas "$ROOT/atlas/pad-top50-atlas.npz" \
  --manifest "$ROOT/manifests/pad-top50-training.jsonl"
touch "$COMPLETE"
echo 'post-stage1 automatic chain complete'
