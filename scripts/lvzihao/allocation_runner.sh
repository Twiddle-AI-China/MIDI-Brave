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

current_experiment_id=
current_action=
write_blocked_on_exit() {
  local status=$?
  local temporary
  trap - EXIT
  if (( status != 0 && status != 75 )) && \
    [[ -n "$current_experiment_id" && ! -f "$queue_state/blocked.txt" ]]; then
    temporary="$queue_state/blocked.${SLURM_JOB_ID}.tmp"
    {
      printf 'failed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'experiment_id=%s\n' "$current_experiment_id"
      printf 'action=%s\n' "$current_action"
      printf 'exit_code=%s\n' "$status"
      printf 'allocation_job_id=%s\n' "$SLURM_JOB_ID"
      printf 'log=%s\n' "$log"
    } >"$temporary"
    mv -- "$temporary" "$queue_state/blocked.txt"
  fi
  exit "$status"
}
trap write_blocked_on_exit EXIT

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

queue_global_initializer=${LV_INITIALIZE_FROM:-}

parse_initializer_action_spec() {
  local value=$1
  [[ "$value" == *'|'* ]] ||
    die "initializer action spec must be ACTION_SPEC|INITIALIZER_RELATIVE"
  ACTION_SPEC=${value%%|*}
  ACTION_INITIALIZER_RELATIVE=${value#*|}
  [[ -n "$ACTION_SPEC" && -n "$ACTION_INITIALIZER_RELATIVE" ]] ||
    die "initializer action spec contains an empty field"
  [[ "$ACTION_INITIALIZER_RELATIVE" != *'|'* ]] ||
    die "initializer action spec must contain exactly one | separator"
}

bind_initializer_contract() {
  local relative=$1 resolved digest source_update
  local key initializer_contract initializer_candidate
  [[ -z "$queue_global_initializer" ]] ||
    die "per-row initializer actions forbid global LV_INITIALIZE_FROM"
  [[ -n "${LV_PITCH_PROBE:-}" ]] ||
    die "per-row MIDI actions require global LV_PITCH_PROBE"
  [[ "$LV_PITCH_PROBE" == "$LV_WORK_ROOT"/* ]] ||
    die "LV_PITCH_PROBE must be below LV_WORK_ROOT"
  require_directory "$LV_PITCH_PROBE"
  require_file "$LV_PITCH_PROBE/qualification.json"
  resolve_persistent_initializer "$relative"
  resolved=$RESOLVED_INITIALIZER_PATH
  digest=$RESOLVED_INITIALIZER_SHA256
  source_update=$RESOLVED_INITIALIZER_UPDATE

  mkdir -p "$queue_state/initializers"
  key=$(printf '%s' "$relative" | sha256sum | awk '{print $1}')
  initializer_contract="$queue_state/initializers/$key.txt"
  initializer_candidate="$queue_state/initializers/$key.${SLURM_JOB_ID}.tmp"
  {
    printf 'relative_path=%s\n' "$relative"
    printf 'resolved_path=%s\n' "$resolved"
    printf 'sha256=%s\n' "$digest"
    printf 'source_update=%s\n' "$source_update"
  } >"$initializer_candidate"
  if [[ -f "$initializer_contract" ]]; then
    if ! cmp -s "$initializer_candidate" "$initializer_contract"; then
      mv -- "$initializer_candidate" \
        "$queue_state/initializers/$key.mismatch-${SLURM_JOB_ID}.txt"
      die "initializer path/hash contract changed: $relative"
    fi
    rm -- "$initializer_candidate"
  else
    mv -- "$initializer_candidate" "$initializer_contract"
  fi
  BOUND_INITIALIZER_PATH=$resolved
  BOUND_INITIALIZER_SHA256=$digest
  BOUND_INITIALIZER_UPDATE=$source_update
  action_initializer_relative=$relative
  action_initializer_sha256=$digest
}

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
  current_experiment_id=$experiment_id
  current_action=$action
  printf 'experiment_start=%s id=%s action=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment_id" "$action"
  action_initializer_relative=
  action_initializer_sha256=

  set +e
  case "$action" in
    taxonomy)
      [[ "$config_relative" == - && "$run_relative" == - ]] ||
        die "taxonomy config/run fields must be -"
      if [[ "$spec" == - ]]; then
        "$script_dir/taxonomy.sh"
      else
        "$script_dir/taxonomy.sh" "$spec"
      fi
      status=$?
      ;;
    taxonomy_validate_audio)
      [[ "$config_relative" == - && "$run_relative" == - ]] ||
        die "taxonomy_validate_audio config/run fields must be -"
      if [[ "$spec" == - ]]; then
        "$script_dir/taxonomy_validate_audio.sh"
      else
        "$script_dir/taxonomy_validate_audio.sh" "$spec"
      fi
      status=$?
      ;;
    clap_report)
      [[ "$config_relative" == - && "$run_relative" == - ]] ||
        die "clap_report config/run fields must be -"
      [[ "$spec" != - ]] || die "clap_report spec must be an audition id"
      "$script_dir/clap_monitor.sh" "$experiment_id" "$spec"
      status=$?
      ;;
    smoke)
      [[ "$spec" == - ]] || die "smoke spec must be -"
      "$script_dir/smoke.sh" \
        "$config_relative" "$run_relative" "$experiment_id"
      status=$?
      ;;
    midi_smoke_from)
      parse_initializer_action_spec "$spec"
      [[ "$ACTION_SPEC" == - ]] || die "midi_smoke_from action spec must be -"
      bind_initializer_contract "$ACTION_INITIALIZER_RELATIVE"
      LV_INITIALIZE_FROM="$BOUND_INITIALIZER_PATH" \
        LV_EXPECTED_INITIALIZER_SHA256="$BOUND_INITIALIZER_SHA256" \
        LV_INITIALIZER_UPDATE="$BOUND_INITIALIZER_UPDATE" \
        "$script_dir/smoke.sh" \
        "$config_relative" "$run_relative" "$experiment_id"
      status=$?
      ;;
    sweep)
      "$script_dir/sweep.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    midi_sweep_from)
      parse_initializer_action_spec "$spec"
      bind_initializer_contract "$ACTION_INITIALIZER_RELATIVE"
      LV_INITIALIZE_FROM="$BOUND_INITIALIZER_PATH" \
        LV_EXPECTED_INITIALIZER_SHA256="$BOUND_INITIALIZER_SHA256" \
        LV_INITIALIZER_UPDATE="$BOUND_INITIALIZER_UPDATE" \
        "$script_dir/sweep.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$ACTION_SPEC"
      status=$?
      ;;
    train)
      "$script_dir/train.sh" "$config_relative" "$run_relative" "$spec"
      status=$?
      ;;
    midi_train_from)
      parse_initializer_action_spec "$spec"
      bind_initializer_contract "$ACTION_INITIALIZER_RELATIVE"
      LV_INITIALIZE_FROM="$BOUND_INITIALIZER_PATH" \
        LV_EXPECTED_INITIALIZER_SHA256="$BOUND_INITIALIZER_SHA256" \
        LV_INITIALIZER_UPDATE="$BOUND_INITIALIZER_UPDATE" \
        "$script_dir/train.sh" \
        "$config_relative" "$run_relative" "$ACTION_SPEC"
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
    audition_report)
      LV_AUDITION_FAIL_ON_REJECT=0 "$script_dir/audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    midi_audition)
      "$script_dir/midi_audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    midi_audition_report)
      LV_MIDI_AUDITION_FAIL_ON_REJECT=0 "$script_dir/midi_audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$spec"
      status=$?
      ;;
    midi_audition_from|midi_audition_report_from)
      parse_initializer_action_spec "$spec"
      bind_initializer_contract "$ACTION_INITIALIZER_RELATIVE"
      fail_on_reject=1
      [[ "$action" == midi_audition_from ]] || fail_on_reject=0
      LV_INITIALIZE_FROM="$BOUND_INITIALIZER_PATH" \
        LV_EXPECTED_INITIALIZER_SHA256="$BOUND_INITIALIZER_SHA256" \
        LV_INITIALIZER_UPDATE="$BOUND_INITIALIZER_UPDATE" \
        LV_MIDI_AUDITION_FAIL_ON_REJECT="$fail_on_reject" \
        "$script_dir/midi_audition.sh" \
        "$config_relative" "$run_relative" "$experiment_id" "$ACTION_SPEC"
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
    if [[ -n "$action_initializer_relative" ]]; then
      printf 'initializer_relative=%s\n' "$action_initializer_relative"
      printf 'initializer_sha256=%s\n' "$action_initializer_sha256"
    fi
  } >"$done_temporary"
  mv -- "$done_temporary" "$done_marker"
  current_experiment_id=
  current_action=
  printf 'experiment_complete=%s id=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment_id"
done <"$queue"

printf 'completed_at=%s\nqueue=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$queue" \
  >"$queue_state/complete.txt"
printf 'queue_complete=%s\n' "$queue_name"
