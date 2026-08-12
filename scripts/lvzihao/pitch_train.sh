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
(( max_updates <= 10000 )) ||
  die "MAX_UPDATES cannot exceed the pitch trainer limit of 10000"
assert_config_contract "$config_relative" "$run_relative"

: "${LV_PITCH_BATCH_PER_GPU:=64}"
[[ "$LV_PITCH_BATCH_PER_GPU" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_PITCH_BATCH_PER_GPU must be a positive integer"

host_config=$(host_config_path "$config_relative")
config=$(container_config_path "$config_relative")
latent_dim=$(awk '$1 == "latent_dim:" {print $2; exit}' "$host_config")
[[ "$latent_dim" == 128 ]] ||
  die "pitch_train requires a 128D model config, got: ${latent_dim:-missing}"
# zrave_pitch_train derives this sibling path from train.output_root.
pitch_host="$LV_WORK_ROOT/$(dirname -- "$run_relative")/pitch-probe"
checkpoint_host="$pitch_host/checkpoints"
qualification_host="$pitch_host/qualification.json"
pack_index_host="$LV_WORK_ROOT/$LV_PACK_RELATIVE/index.json"
require_file "$pack_index_host"
mkdir -p "$checkpoint_host" "$pitch_host/tensorboard"

verify_qualification() {
  python3 - "$qualification_host" "$pitch_host" \
    "$host_config" "$pack_index_host" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

qualification_path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()

def digest(path: str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

try:
    report = json.loads(qualification_path.read_text(encoding="utf-8"))
    checkpoint_value = report.get("checkpoint")
    checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    valid = (
        report.get("passed") is True
        and checkpoint is not None
        and checkpoint.is_file()
        and checkpoint.resolve().is_relative_to(root)
        and report.get("config_sha256") == digest(sys.argv[3])
        and report.get("pack_index_sha256") == digest(sys.argv[4])
    )
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

if [[ -f "$qualification_host" ]] && verify_qualification; then
  printf 'pitch probe already qualified: %s\n' "$qualification_host"
  exit 0
fi

resume_args=()
if latest=$(latest_checkpoint "$checkpoint_host"); then
  resume_container=$(host_path_to_container_work_path "$latest")
  resume_args=(--resume "$resume_container")
fi

printf 'pitch_train config=%s batch=%s max_updates=%s resume=%s\n' \
  "$config_relative" "$LV_PITCH_BATCH_PER_GPU" "$max_updates" \
  "${resume_args[*]:-none}"

set +e
run_timed_gpu_container \
  python -m midibrave.zrave_pitch_train \
    --config "$config" \
    --batch-per-gpu "$LV_PITCH_BATCH_PER_GPU" \
    --max-updates "$max_updates" \
    "${resume_args[@]}"
status=$?
set -e

if (( status == 75 )); then
  if latest=$(latest_checkpoint "$checkpoint_host"); then
    printf 'allocation boundary reached; next worker resumes %s\n' "$latest"
  else
    printf 'allocation boundary reached before the first pitch checkpoint\n' >&2
  fi
  exit 75
fi
(( status == 0 )) || exit "$status"

require_file "$qualification_host"
verify_qualification ||
  die "pitch training exited successfully without a passed qualification/checkpoint"
printf 'pitch_qualification=%s\n' "$qualification_host"
