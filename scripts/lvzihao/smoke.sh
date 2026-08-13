#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 3 )); then
  printf 'usage: %s CONFIG_RELATIVE RUN_RELATIVE EXPERIMENT_ID\n' "$0" >&2
  exit 2
fi

config_relative=$1
run_relative=$2
experiment_id=$3
assert_config_contract "$config_relative" "$run_relative"
[[ "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid experiment id: $experiment_id"
config=$(container_config_path "$config_relative")
optional_pitch_probe_args
host_config=$(host_config_path "$config_relative")
model_profile=$(awk '$1 == "profile:" {print $2; exit}' "$host_config")
model_profile=${model_profile:-standard}
pitch_conditioning=$(awk '$1 == "pitch_conditioning:" {print $2; exit}' "$host_config")
pitch_conditioning=${pitch_conditioning:-true}
if [[ "$model_profile" != standard && "$pitch_conditioning" == false ]]; then
  [[ -z "${LV_INITIALIZE_FROM:-}" ]] ||
    die "non-standard pure profiles train from scratch; unset LV_INITIALIZE_FROM"
elif (
  [[ "$model_profile" != standard ]] &&
  [[ "$pitch_conditioning" == true ]] &&
  [[ -z "${LV_INITIALIZE_FROM:-}" ]]
); then
  die "non-standard MIDI profiles require a same-profile pure LV_INITIALIZE_FROM"
fi
INITIALIZER_ARGS=()
if [[ -n "${LV_INITIALIZE_FROM:-}" ]]; then
  verify_initializer_environment "$LV_INITIALIZE_FROM"
  initializer_container=$(
    host_path_to_container_work_path "$LV_INITIALIZE_FROM"
  )
  INITIALIZER_ARGS=(--initialize-from "$initializer_container")
fi

mkdir -p "$LV_WORK_ROOT/smoke"
report_host="$LV_WORK_ROOT/smoke/${experiment_id}-${SLURM_JOB_ID}.json"
report_container=$(host_path_to_container_work_path "$report_host")

run_timed_gpu_container python -c '
import json
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
properties = torch.cuda.get_device_properties(0)
assert properties.major >= 12, properties
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
value = torch.randn(256, 256, device="cuda")
result = value @ value.T
assert torch.isfinite(result).all()
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": properties.name,
    "capability": list(torch.cuda.get_device_capability(0)),
    "device_count": torch.cuda.device_count(),
}, sort_keys=True))
'

run_timed_gpu_container \
  python -m midibrave.zrave_flow_train \
    --config "$config" \
    --batch-per-gpu 5 \
    --benchmark-report "$report_container" \
    --benchmark-warmup 1 \
    --benchmark-updates 2 \
    --benchmark-exposure-updates 5 \
    "${INITIALIZER_ARGS[@]}" \
    "${PITCH_PROBE_ARGS[@]}"

require_file "$report_host"
printf 'smoke_report=%s\n' "$report_host"
