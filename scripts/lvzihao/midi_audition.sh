#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 4 )); then
  printf 'usage: %s CONFIG_RELATIVE RUN_RELATIVE EXPERIMENT_ID CHECKPOINT\n' "$0" >&2
  exit 2
fi

config_relative=$1
run_relative=$2
experiment_id=$3
checkpoint_spec=$4
assert_config_contract "$config_relative" "$run_relative"
[[ "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid experiment id: $experiment_id"
[[ -n "${LV_SOURCE_ROOT:-}" ]] || die "LV_SOURCE_ROOT is required"
[[ -n "${LV_PITCH_PROBE:-}" ]] || die "LV_PITCH_PROBE is required"
[[ -n "${LV_INITIALIZE_FROM:-}" ]] || die "LV_INITIALIZE_FROM is required"
require_directory "$LV_SOURCE_ROOT"
require_directory "$LV_PITCH_PROBE"
require_file "$LV_PITCH_PROBE/qualification.json"
require_file "$LV_INITIALIZE_FROM"
require_command sha256sum

config=$(container_config_path "$config_relative")
checkpoint_root="$LV_WORK_ROOT/$run_relative/checkpoints"
if [[ "$checkpoint_spec" == latest ]]; then
  checkpoint=$(latest_checkpoint "$checkpoint_root") ||
    die "no step checkpoint exists in $checkpoint_root"
else
  [[ "$checkpoint_spec" =~ ^(step-[0-9]+|final)\.pt$ ]] ||
    die "checkpoint must be latest, final.pt, or a step-N.pt basename"
  checkpoint="$checkpoint_root/$checkpoint_spec"
fi
require_file "$checkpoint"
checkpoint_container=$(host_path_to_container_work_path "$checkpoint")
pitch_container=$(host_path_to_container_work_path "$LV_PITCH_PROBE")
checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
initializer_sha=$(sha256sum "$LV_INITIALIZE_FROM" | awk '{print $1}')

: "${LV_AUDITION_GENERATED_FRAMES:=320}"
[[ "$LV_AUDITION_GENERATED_FRAMES" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_AUDITION_GENERATED_FRAMES must be positive"

audition_root="$LV_WORK_ROOT/auditions"
output="$audition_root/$experiment_id"
if [[ -f "$output/manifest.json" && -f "$output/gate.json" ]]; then
  if python3 - "$output/manifest.json" "$output/gate.json" \
    "$checkpoint_sha" "$initializer_sha" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
valid = (
    manifest.get("checkpoint", {}).get("sha256") == sys.argv[3]
    and manifest.get("checkpoint", {}).get("initialization", {}).get(
        "checkpoint_sha256"
    ) == sys.argv[4]
    and gate.get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
  then
    printf 'MIDI audition already passed degeneration gate: %s\n' "$output"
    exit 0
  fi
  die "MIDI audition output belongs to another contract or failed gate: $output"
fi
[[ ! -e "$output" ]] || die "MIDI audition output is incomplete: $output"

partial="$audition_root/.partial-${experiment_id}-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || die "partial output already exists: $partial"
mkdir -p "$partial"
partial_container=$(host_path_to_container_work_path "$partial")

set +e
run_timed_gpu_container \
  python scripts/render_zrave_midi_flow_audition.py \
    --config "$config" \
    --checkpoint "$checkpoint_container" \
    --expected-checkpoint-sha256 "$checkpoint_sha" \
    --expected-initializer-sha256 "$initializer_sha" \
    --pitch-qualification "$pitch_container/qualification.json" \
    --output "$partial_container" \
    --categories Pad Lead Bass Pluck Keys Synth \
    --note-vocabulary 36 62 82 \
    --generation-seeds 17 71 \
    --generated-frames "$LV_AUDITION_GENERATED_FRAMES" \
    --device cuda
status=$?
set -e
(( status == 0 )) || exit "$status"

# This is the generic 128D long-rollout degeneration gate. It does not yet
# measure whether the generated pitch follows matched/note-swap MIDI control.
run_timed_gpu_container \
  python -m midibrave.zrave_flow_gate \
    --manifest "$partial_container/manifest.json" \
    --output "$partial_container/gate.json" \
    --fail-on-reject

mv -- "$partial" "$output"
printf 'midi_audition=%s gate=%s\n' "$output" "$output/gate.json"
