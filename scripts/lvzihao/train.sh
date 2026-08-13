#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 3 )); then
  printf 'usage: %s CONFIG_RELATIVE RUN_RELATIVE MAX_UPDATES\n' "$0" >&2
  exit 2
fi

config_relative=$1
run_relative=$2
max_updates=$3
[[ "$max_updates" =~ ^[1-9][0-9]*$ ]] ||
  die "MAX_UPDATES must be a positive integer"
assert_config_contract "$config_relative" "$run_relative"

host_config=$(host_config_path "$config_relative")
config=$(container_config_path "$config_relative")
run_host="$LV_WORK_ROOT/$run_relative"
checkpoint_host="$run_host/checkpoints"
selection_host="$run_host/lvzihao-selection.json"
pack_index_host="$LV_WORK_ROOT/$LV_PACK_RELATIVE/index.json"
mkdir -p "$checkpoint_host" "$run_host/tensorboard"
require_file "$pack_index_host"

if [[ -n "${LV_BATCH_PER_GPU:-}" ]]; then
  batch=$LV_BATCH_PER_GPU
  [[ "$batch" =~ ^[1-9][0-9]*$ ]] ||
    die "LV_BATCH_PER_GPU must be a positive integer"
else
  require_file "$selection_host"
  batch=$(python3 "$script_dir/select_sweep.py" print-verified-batch \
    --selection "$selection_host" \
    --config "$host_config" \
    --pack-index "$pack_index_host" \
    --project-root "$LV_PROJECT_ROOT")
fi

resume_args=()
if latest=$(latest_checkpoint "$checkpoint_host"); then
  resume_container=$(host_path_to_container_work_path "$latest")
  resume_args=(--resume "$resume_container")
elif [[ -f "$checkpoint_host/final.pt" ]]; then
  final_container=$(
    host_path_to_container_work_path "$checkpoint_host/final.pt"
  )
  resume_args=(--resume "$final_container")
fi

initialize_args=()
expected_initializer_args=()
segment_enabled=$(awk '$1 == "enabled:" {print $2; exit}' "$host_config")
model_profile=$(awk '$1 == "profile:" {print $2; exit}' "$host_config")
model_profile=${model_profile:-standard}
pitch_conditioning=$(awk '$1 == "pitch_conditioning:" {print $2; exit}' "$host_config")
pitch_conditioning=${pitch_conditioning:-true}
if [[ "$model_profile" != standard && "$pitch_conditioning" == false ]]; then
  [[ -z "${LV_INITIALIZE_FROM:-}" ]] ||
    die "non-standard pure profiles train from scratch; unset LV_INITIALIZE_FROM"
fi
if ((${#resume_args[@]} == 0)) && [[ -z "${LV_INITIALIZE_FROM:-}" ]]; then
  if [[ "$model_profile" == standard && "$segment_enabled" == true ]]; then
    die "new segment runs require LV_INITIALIZE_FROM for weight-only initialization"
  fi
  if [[ "$model_profile" != standard && "$pitch_conditioning" == true ]]; then
    die "non-standard MIDI runs require a same-profile pure LV_INITIALIZE_FROM"
  fi
fi
if [[ -n "${LV_INITIALIZE_FROM:-}" && ${#resume_args[@]} -eq 0 ]]; then
  verify_initializer_environment "$LV_INITIALIZE_FROM"
  initialize_container=$(
    host_path_to_container_work_path "$LV_INITIALIZE_FROM"
  )
  initialize_args=(--initialize-from "$initialize_container")
elif [[ -n "${LV_INITIALIZE_FROM:-}" ]]; then
  # A resumed run no longer reloads the initializer, but its immutable
  # same-family lineage is still checked on every allocation.
  verify_initializer_environment "$LV_INITIALIZE_FROM"
fi
if [[ -n "${LV_INITIALIZE_FROM:-}" ]]; then
  expected_initializer_sha256=${LV_EXPECTED_INITIALIZER_SHA256:-$INITIALIZER_SHA256}
  expected_initializer_update=${LV_INITIALIZER_UPDATE:-}
  if [[ -z "$expected_initializer_update" ]]; then
    initializer_basename=$(basename -- "$LV_INITIALIZE_FROM")
    [[ "$initializer_basename" =~ ^step-([0-9]+)\.pt$ ]] ||
      die "LV_INITIALIZER_UPDATE is required unless initializer is step-N.pt"
    expected_initializer_update=$((10#${BASH_REMATCH[1]}))
  fi
  [[ "$expected_initializer_update" =~ ^[1-9][0-9]*$ ]] ||
    die "LV_INITIALIZER_UPDATE must be a positive integer"
  expected_initializer_args=(
    --expected-initializer-sha256 "$expected_initializer_sha256"
    --expected-initializer-update "$expected_initializer_update"
  )
elif [[ -n "${LV_EXPECTED_INITIALIZER_SHA256:-}" ]] ||
  [[ -n "${LV_INITIALIZER_UPDATE:-}" ]]; then
  die "initializer SHA/update require LV_INITIALIZE_FROM"
fi
optional_pitch_probe_args

printf 'train config=%s batch=%s max_updates=%s resume=%s initialize=%s\n' \
  "$config_relative" "$batch" "$max_updates" \
  "${resume_args[*]:-none}" "${initialize_args[*]:-none}"

set +e
run_timed_gpu_container \
  python -m midibrave.zrave_flow_train \
    --config "$config" \
    --batch-per-gpu "$batch" \
    --max-updates "$max_updates" \
    "${resume_args[@]}" \
    "${initialize_args[@]}" \
    "${expected_initializer_args[@]}" \
    "${PITCH_PROBE_ARGS[@]}"
status=$?
set -e

if (( status == 75 )); then
  if latest=$(latest_checkpoint "$checkpoint_host"); then
    printf 'allocation boundary reached; next worker resumes %s\n' "$latest"
  else
    printf 'allocation boundary reached before the first checkpoint\n' >&2
  fi
  exit 75
fi
(( status == 0 )) || exit "$status"

require_file "$checkpoint_host/final.pt"
printf 'final_checkpoint=%s\n' "$checkpoint_host/final.pt"
