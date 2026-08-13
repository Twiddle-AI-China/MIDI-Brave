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
host_config=$(host_config_path "$config_relative")
load_single_source_allowed_categories "$host_config"
[[ "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid experiment id: $experiment_id"
[[ -n "${LV_SOURCE_ROOT:-}" ]] || die "LV_SOURCE_ROOT is required"
[[ -n "${LV_PITCH_PROBE:-}" ]] || die "LV_PITCH_PROBE is required"
[[ -n "${LV_INITIALIZE_FROM:-}" ]] || die "LV_INITIALIZE_FROM is required"
require_directory "$LV_SOURCE_ROOT"
require_directory "$LV_PITCH_PROBE"
require_file "$LV_PITCH_PROBE/qualification.json"
require_command sha256sum
verify_initializer_environment "$LV_INITIALIZE_FROM"

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
initializer_sha=$INITIALIZER_SHA256

: "${LV_AUDITION_GENERATED_FRAMES:=320}"
: "${LV_INITIALIZER_UPDATE:=85000}"
: "${LV_MIDI_AUDITION_FAIL_ON_REJECT:=1}"
: "${LV_AUDIO_EVAL_IMAGE:=midibrave:lvzihao-cu128-audio-eval-v1}"
: "${LV_CREPE_MODEL_SHA256:=d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432}"
[[ "$LV_AUDITION_GENERATED_FRAMES" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_AUDITION_GENERATED_FRAMES must be positive"
[[ "$LV_INITIALIZER_UPDATE" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_INITIALIZER_UPDATE must be positive"
[[ "$LV_MIDI_AUDITION_FAIL_ON_REJECT" =~ ^[01]$ ]] ||
  die "LV_MIDI_AUDITION_FAIL_ON_REJECT must be 0 or 1"
[[ "$LV_CREPE_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "LV_CREPE_MODEL_SHA256 must be a lowercase SHA-256"

audition_root="$LV_WORK_ROOT/auditions"
output="$audition_root/$experiment_id"
if [[ -f "$output/manifest.json" && -f "$output/gate.json" \
  && -f "$output/midi-gate.json" \
  && -f "$output/decoded-audio-midi-gate.json" \
  && -f "$output/qualification.json" ]]; then
  if python3 - "$output/manifest.json" "$output/gate.json" \
    "$output/midi-gate.json" "$output/decoded-audio-midi-gate.json" \
    "$output/qualification.json" \
    "$checkpoint_sha" "$initializer_sha" \
    "$LV_INITIALIZER_UPDATE" "$LV_MIDI_AUDITION_FAIL_ON_REJECT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
gate_paths = {
    "generic_long_rollout": Path(sys.argv[2]),
    "latent_pitch_probe_proxy": Path(sys.argv[3]),
    "decoded_audio_crepe": Path(sys.argv[4]),
}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
gate = json.loads(gate_paths["generic_long_rollout"].read_text(encoding="utf-8"))
midi_gate = json.loads(
    gate_paths["latent_pitch_probe_proxy"].read_text(encoding="utf-8")
)
decoded_gate = json.loads(
    gate_paths["decoded_audio_crepe"].read_text(encoding="utf-8")
)
qualification = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
hard = sys.argv[9] == "1"
gate_results_valid = (
    isinstance(gate.get("passed"), bool)
    and isinstance(midi_gate.get("passed"), bool)
    and isinstance(decoded_gate.get("passed"), bool)
    and decoded_gate.get("metric_kind") == "decoded_audio_crepe"
    and (not hard or gate["passed"] is True)
    and (not hard or midi_gate["passed"] is True)
    and (not hard or decoded_gate["passed"] is True)
)
all_passed = all(
    value.get("passed") is True for value in (gate, midi_gate, decoded_gate)
)
gate_results = {
    "generic_long_rollout": gate.get("passed") is True,
    "latent_pitch_probe_proxy": midi_gate.get("passed") is True,
    "decoded_audio_crepe": decoded_gate.get("passed") is True,
}
gate_hashes = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in gate_paths.items()
}
valid = (
    manifest.get("checkpoint", {}).get("sha256") == sys.argv[6]
    and manifest.get("checkpoint", {}).get("initialization", {}).get(
        "checkpoint_sha256"
    ) == sys.argv[7]
    and manifest.get("checkpoint", {}).get("initialization", {}).get(
        "source_update"
    ) == int(sys.argv[8])
    and gate.get("manifest_sha256") == manifest_sha256
    and midi_gate.get("manifest_sha256") == manifest_sha256
    and decoded_gate.get("manifest_sha256") == manifest_sha256
    and decoded_gate.get("lineage", {}).get(
        "audition_manifest_sha256"
    ) == manifest_sha256
    and gate_results_valid
    and qualification.get("schema") == 1
    and qualification.get("qualified") is False
    and qualification.get("midi_adherence_calibrated") is False
    and qualification.get("checkpoint_sha256") == sys.argv[6]
    and qualification.get("gates") == gate_results
    and qualification.get("gate_sha256") == gate_hashes
    and qualification.get("qualification_status")
    == (
        "provisional_gates_passed_uncalibrated"
        if hard and all_passed
        else "research_only"
    )
)
raise SystemExit(0 if valid else 1)
PY
  then
    printf 'MIDI audition already complete: %s\n' "$output"
    exit 0
  fi
  die "MIDI audition output belongs to another contract or failed gate: $output"
fi
[[ ! -e "$output" ]] || die "MIDI audition output is incomplete: $output"

# Training remains on the lean image. Decoded-audio MIDI qualification is a
# separate offline runtime whose official torchcrepe model asset is baked into
# the image and SHA-verified by the evaluator.
require_command docker
docker image inspect "$LV_AUDIO_EVAL_IMAGE" >/dev/null 2>&1 ||
  die "decoded_audio_crepe blocked: build audited image with scripts/lvzihao/build_audio_eval_image.sh"

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
    --expected-initializer-update "$LV_INITIALIZER_UPDATE" \
    --pitch-qualification "$pitch_container/qualification.json" \
    --output "$partial_container" \
    --categories "${ALLOWED_CATEGORIES[@]}" \
    --note-vocabulary 36 62 82 \
    --generation-seeds 17 71 \
    --generated-frames "$LV_AUDITION_GENERATED_FRAMES" \
    --device cuda
status=$?
set -e
(( status == 0 )) || exit "$status"

flow_gate_reject_arg=--fail-on-reject
midi_gate_reject_args=(--fail-on-reject)
decoded_gate_reject_args=(--fail-on-reject)
if [[ "$LV_MIDI_AUDITION_FAIL_ON_REJECT" == 0 ]]; then
  flow_gate_reject_arg=--no-fail-on-reject
  midi_gate_reject_args=()
  decoded_gate_reject_args=()
fi

# All three reports are always materialized. Research mode only softens a
# negative threshold decision; rendering, lineage/hash checks, and report
# integrity stay hard failures.
run_timed_gpu_container \
  python -m midibrave.zrave_flow_gate \
    --manifest "$partial_container/manifest.json" \
    --output "$partial_container/gate.json" \
    "$flow_gate_reject_arg"

run_timed_gpu_container \
  python -m midibrave.zrave_midi_adherence_gate \
    --manifest "$partial_container/manifest.json" \
    --output "$partial_container/midi-gate.json" \
    "${midi_gate_reject_args[@]}"

(
  export LV_IMAGE="$LV_AUDIO_EVAL_IMAGE"
  run_timed_gpu_container \
    python -m midibrave.zrave_decoded_audio_midi_gate \
      --manifest "$partial_container/manifest.json" \
      --output "$partial_container/decoded-audio-midi-gate.json" \
      --device cuda \
      --expected-model-sha256 "$LV_CREPE_MODEL_SHA256" \
      "${decoded_gate_reject_args[@]}"
)

python3 - "$partial/manifest.json" "$partial/gate.json" \
  "$partial/midi-gate.json" "$partial/decoded-audio-midi-gate.json" \
  "$LV_MIDI_AUDITION_FAIL_ON_REJECT" "$partial/qualification.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, generic_path, latent_path, decoded_path = map(Path, sys.argv[1:5])
hard = sys.argv[5] == "1"
output = Path(sys.argv[6])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
paths = {
    "generic_long_rollout": generic_path,
    "latent_pitch_probe_proxy": latent_path,
    "decoded_audio_crepe": decoded_path,
}
gates = {
    name: json.loads(path.read_text(encoding="utf-8")).get("passed") is True
    for name, path in paths.items()
}
all_passed = all(gates.values())
# Decoded-audio thresholds have not yet been calibrated against the same
# source/RAVE-direct held-out ceiling. Preserve hard gate results, but do not
# mint a production "qualified" claim until that calibration contract exists.
status = (
    "provisional_gates_passed_uncalibrated"
    if hard and all_passed
    else "research_only"
)
payload = {
    "schema": 1,
    "qualification_status": status,
    "qualified": False,
    "evaluation_mode": "hard" if hard else "research_report",
    "midi_adherence_calibrated": False,
    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
    "gates": gates,
    "gate_sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    },
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
PY

mv -- "$partial" "$output"
printf 'midi_audition=%s gate=%s midi_gate=%s decoded_audio_midi_gate=%s\n' \
  "$output" "$output/gate.json" "$output/midi-gate.json" \
  "$output/decoded-audio-midi-gate.json"
