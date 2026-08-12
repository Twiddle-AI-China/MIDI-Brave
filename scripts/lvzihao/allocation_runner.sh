#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 1 )); then
  printf 'usage: %s QUEUE_TSV\n' "$0" >&2
  exit 2
fi

queue=$1
require_file "$queue"
assert_persistent_work_root
assert_single_gpu_allocation
assert_clean_checkout
require_command flock
require_command sha256sum
require_command docker
require_command cmp
require_command tee

: "${LV_ALLOCATION_BUDGET_SECONDS:=252000}"
[[ "$LV_ALLOCATION_BUDGET_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_ALLOCATION_BUDGET_SECONDS must be a positive integer"
(( LV_ALLOCATION_BUDGET_SECONDS <= 72 * 60 * 60 )) ||
  die "allocation budget cannot exceed 72 hours"
export LV_DEADLINE_EPOCH=$(( $(date +%s) + LV_ALLOCATION_BUDGET_SECONDS ))

queue_name=$(basename -- "$queue")
queue_name=${queue_name%.tsv}
[[ "$queue_name" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "queue basename is unsafe: $queue_name"
queue_state="$LV_STATE_ROOT/queues/$queue_name"
mkdir -p "$queue_state/done" "$queue_state/logs"

exec 9>"$queue_state/runner.lock"
if ! flock -w 300 9; then
  die "another allocation owns queue $queue_name"
fi

log="$queue_state/logs/allocation-${SLURM_JOB_ID}.log"
exec > >(tee -a "$log") 2>&1
printf 'allocation_start=%s job_id=%s gpu=%s queue=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" \
  "$CUDA_VISIBLE_DEVICES" "$queue"

queue_sha=$(sha256sum "$queue" | awk '{print $1}')
git_commit=$(git -C "$LV_PROJECT_ROOT" rev-parse HEAD)
image_id=$(docker image inspect --format '{{.Id}}' "$LV_IMAGE")
contract="$queue_state/contract.txt"
contract_candidate="$queue_state/contract.${SLURM_JOB_ID}.tmp"
{
  printf 'queue_sha256=%s\n' "$queue_sha"
  printf 'git_commit=%s\n' "$git_commit"
  printf 'image=%s\n' "$LV_IMAGE"
  printf 'image_id=%s\n' "$image_id"
  printf 'host_repo_root=%s\n' "$LV_PROJECT_ROOT"
  printf 'host_flow_root=%s\n' "$LV_WORK_ROOT"
  printf 'container_flow_root=%s\n' "$LV_CONTAINER_WORK_ROOT"
} >"$contract_candidate"
if [[ -f "$contract" ]]; then
  if ! cmp -s "$contract_candidate" "$contract"; then
    printf 'queue contract changed; expected:\n' >&2
    sed 's/^/  /' "$contract" >&2
    printf 'observed:\n' >&2
    sed 's/^/  /' "$contract_candidate" >&2
    mv -- "$contract_candidate" "$queue_state/contract-mismatch-${SLURM_JOB_ID}.txt"
    exit 2
  fi
  rm -- "$contract_candidate"
else
  mv -- "$contract_candidate" "$contract"
fi

if [[ -f "$queue_state/blocked.txt" ]]; then
  printf 'queue is blocked; inspect %s\n' "$queue_state/blocked.txt" >&2
  exit 1
fi
if [[ -f "$queue_state/complete.txt" ]]; then
  printf 'queue already complete: %s\n' "$queue_name"
  exit 0
fi

declare -A observed_ids=()
line_number=0
while IFS=$'\t' read -r experiment_id action config_relative \
  run_relative spec extra || [[ -n "${experiment_id:-}" ]]; do
  line_number=$((line_number + 1))
  [[ -n "${experiment_id:-}" ]] || continue
  [[ "$experiment_id" == \#* ]] && continue
  [[ -z "${extra:-}" ]] ||
    die "queue line $line_number has more than five tab-separated fields"
  [[ "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
    die "unsafe experiment id on queue line $line_number"
  [[ -z "${observed_ids[$experiment_id]:-}" ]] ||
    die "duplicate experiment id: $experiment_id"
  observed_ids[$experiment_id]=1
  [[ -n "$action" && -n "$config_relative" && -n "$run_relative" ]] ||
    die "queue line $line_number has an empty required field"
  [[ -n "$spec" ]] || die "queue line $line_number has an empty spec"

  done_marker="$queue_state/done/$experiment_id.txt"
  [[ ! -f "$done_marker" ]] || continue
  printf 'experiment_start=%s id=%s action=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment_id" "$action"

  set +e
  case "$action" in
    smoke)
      [[ "$spec" == - ]] || die "smoke spec must be -"
      "$script_dir/smoke.sh" \
        "$config_relative" "$run_relative" "$experiment_id"
      status=$?
      ;;
    sweep)
      "$script_dir/sweep.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    train)
      "$script_dir/train.sh" "$config_relative" "$run_relative" "$spec"
      status=$?
      ;;
    pitch_train)
      "$script_dir/pitch_train.sh" \
        "$config_relative" "$run_relative" "$spec"
      status=$?
      ;;
    audition)
      "$script_dir/audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    midi_audition)
      "$script_dir/midi_audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    *)
      printf 'unknown action on queue line %s: %s\n' \
        "$line_number" "$action" >&2
      status=2
      ;;
  esac
  set -e

  if (( status == 75 )); then
    printf 'allocation_continuation_required=%s id=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment_id"
    exit 75
  fi
  if (( status != 0 )); then
    {
      printf 'failed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'experiment_id=%s\n' "$experiment_id"
      printf 'action=%s\n' "$action"
      printf 'exit_code=%s\n' "$status"
      printf 'allocation_job_id=%s\n' "$SLURM_JOB_ID"
      printf 'log=%s\n' "$log"
    } >"$queue_state/blocked.txt"
    exit "$status"
  fi

  done_temporary="$done_marker.${SLURM_JOB_ID}.tmp"
  {
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'experiment_id=%s\n' "$experiment_id"
    printf 'action=%s\n' "$action"
    printf 'allocation_job_id=%s\n' "$SLURM_JOB_ID"
  } >"$done_temporary"
  mv -- "$done_temporary" "$done_marker"
  printf 'experiment_complete=%s id=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment_id"
done <"$queue"

printf 'completed_at=%s\nqueue=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$queue" \
  >"$queue_state/complete.txt"
printf 'queue_complete=%s\n' "$queue_name"
