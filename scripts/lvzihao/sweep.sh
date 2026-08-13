#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 4 )); then
  printf 'usage: %s CONFIG_RELATIVE RUN_RELATIVE EXPERIMENT_ID BATCHES_CSV\n' "$0" >&2
  exit 2
fi

config_relative=$1
run_relative=$2
experiment_id=$3
batches_csv=$4
assert_config_contract "$config_relative" "$run_relative"
host_config=$(host_config_path "$config_relative")
pack_index_host="$LV_WORK_ROOT/$LV_PACK_RELATIVE/index.json"
require_file "$pack_index_host"
require_command sha256sum
[[ "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid experiment id: $experiment_id"
[[ "$batches_csv" =~ ^[0-9]+(,[0-9]+)*$ ]] ||
  die "batches must be comma-separated positive integers"

IFS=',' read -r -a batches <<<"$batches_csv"
for batch in "${batches[@]}"; do
  (( batch > 0 )) || die "batch values must be positive"
done

: "${LV_BENCHMARK_WARMUP:=10}"
: "${LV_BENCHMARK_UPDATES:=100}"
: "${LV_BENCHMARK_EXPOSURE_UPDATES:=5}"
segment_enabled=$(awk '$1 == "enabled:" {print $2; exit}' "$host_config")
sequence_enabled=$(awk '$1 == "midi_sequence_conditioning:" {print $2; exit}' "$host_config")
expected_exposure_updates=$LV_BENCHMARK_EXPOSURE_UPDATES
if [[ "$segment_enabled" == true || "$sequence_enabled" == true ]]; then
  expected_exposure_updates=0
fi

config=$(container_config_path "$config_relative")
sweep_host="$LV_WORK_ROOT/sweeps/$experiment_id"
sweep_container="$LV_CONTAINER_WORK_ROOT/sweeps/$experiment_id"
selection_host="$LV_WORK_ROOT/$run_relative/lvzihao-selection.json"
mkdir -p "$sweep_host" "$(dirname -- "$selection_host")"
optional_pitch_probe_args
INITIALIZER_ARGS=()
if [[ -n "${LV_INITIALIZE_FROM:-}" ]]; then
  verify_initializer_environment "$LV_INITIALIZE_FROM"
  initializer_container=$(
    host_path_to_container_work_path "$LV_INITIALIZE_FROM"
  )
  INITIALIZER_ARGS=(--initialize-from "$initializer_container")
fi
expected_config_sha=$(sha256sum "$host_config" | awk '{print $1}')
expected_pack_sha=$(sha256sum "$pack_index_host" | awk '{print $1}')
expected_commit=$(git -C "$LV_PROJECT_ROOT" rev-parse HEAD)

report_is_valid() {
  python3 - "$1" "$LV_BENCHMARK_UPDATES" \
    "$expected_exposure_updates" "$expected_config_sha" \
    "$expected_pack_sha" "$expected_commit" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
    valid = (
        value.get("status") == "ok"
        and int(value.get("world_size", 0)) == 1
        and int(value.get("measured_updates", 0)) == int(sys.argv[2])
        and int(value.get("exposure_safety_updates", 0)) == int(sys.argv[3])
        and value.get("config_sha256") == sys.argv[4]
        and value.get("pack_index_sha256") == sys.argv[5]
        and value.get("git_commit") == sys.argv[6]
    )
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

for batch in "${batches[@]}"; do
  report_host="$sweep_host/batch-${batch}.json"
  report_container="$sweep_container/batch-${batch}.json"
  if [[ -f "$report_host" ]] && report_is_valid "$report_host"; then
    printf 'reuse valid sweep report: %s\n' "$report_host"
    continue
  fi

  printf 'benchmark batch_per_gpu=%s\n' "$batch"
  set +e
  run_timed_gpu_container \
    python -m midibrave.zrave_flow_train \
      --config "$config" \
      --batch-per-gpu "$batch" \
      --benchmark-report "$report_container" \
      --benchmark-warmup "$LV_BENCHMARK_WARMUP" \
      --benchmark-updates "$LV_BENCHMARK_UPDATES" \
      --benchmark-exposure-updates "$expected_exposure_updates" \
      "${INITIALIZER_ARGS[@]}" \
      "${PITCH_PROBE_ARGS[@]}"
  status=$?
  set -e
  if (( status == 75 )); then
    exit 75
  fi
  if (( status != 0 )) || [[ ! -f "$report_host" ]]; then
    temporary="$report_host.tmp"
    printf '{"batch_per_gpu":%s,"process_exit_code":%s,"status":"failed","world_size":1}\n' \
      "$batch" "$status" >"$temporary"
    mv -- "$temporary" "$report_host"
  fi
done

python3 "$script_dir/select_sweep.py" select \
  --sweep-root "$sweep_host" \
  --output "$selection_host" \
  --batches "${batches[@]}" \
  --measured-updates "$LV_BENCHMARK_UPDATES" \
  --exposure-updates "$expected_exposure_updates"

printf 'selection=%s\n' "$selection_host"
