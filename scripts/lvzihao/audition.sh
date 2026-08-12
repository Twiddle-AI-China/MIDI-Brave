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

if [[ -z "${LV_SOURCE_ROOT:-}" ]]; then
  die "LV_SOURCE_ROOT is required for source/reference audio audition"
fi
require_directory "$LV_SOURCE_ROOT"

: "${LV_AUDITION_GENERATED_FRAMES:=320}"
: "${LV_AUDITION_CANDIDATES:=2}"
[[ "$LV_AUDITION_GENERATED_FRAMES" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_AUDITION_GENERATED_FRAMES must be positive"
[[ "$LV_AUDITION_CANDIDATES" =~ ^[1-9][0-9]*$ ]] ||
  die "LV_AUDITION_CANDIDATES must be positive"

audition_root="$LV_WORK_ROOT/auditions"
output="$audition_root/$experiment_id"
if [[ -f "$output/verification.json" && -f "$output/gate.json" ]]; then
  if python3 - "$output/verification.json" "$output/gate.json" \
    "$(host_config_path "$config_relative")" "$checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {
    "config_sha256": digest(Path(sys.argv[3])),
    "checkpoint_sha256": digest(Path(sys.argv[4])),
}
valid = gate.get("passed") is True and all(
    report.get(k) == v for k, v in expected.items()
)
raise SystemExit(0 if valid else 1)
PY
  then
    printf 'audition already complete: %s\n' "$output"
    exit 0
  fi
  die "audition output belongs to another config/checkpoint: $output"
fi
[[ ! -e "$output" ]] ||
  die "audition output exists without verification: $output"

partial="$audition_root/.partial-${experiment_id}-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || die "partial output already exists: $partial"
mkdir -p "$partial"
partial_container=$(host_path_to_container_work_path "$partial")

set +e
run_timed_gpu_container \
  python scripts/render_zrave_flow_audition.py \
    --config "$config" \
    --checkpoint "$checkpoint_container" \
    --output "$partial_container" \
    --categories Pad Lead Bass Pluck Keys Synth \
    --explorations 0.0 0.5 1.0 \
    --generation-seeds 17 71 \
    --generated-frames "$LV_AUDITION_GENERATED_FRAMES" \
    --candidate-count "$LV_AUDITION_CANDIDATES" \
    --selection-seed 20260802 \
    --device cuda
status=$?
set -e
(( status == 0 )) || exit "$status"

run_timed_gpu_container \
  python - "$partial_container" "$config" "$checkpoint_container" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
checkpoint_sha256 = digest(sys.argv[3])
if manifest["checkpoint"]["sha256"] != checkpoint_sha256:
    raise ValueError("audition manifest checkpoint hash mismatch")
paths = []
for example in manifest.get("examples", []):
    takes = [example["source"], example["direct"], *example["rollouts"]]
    for take in takes:
        paths.extend((take["raw_wav"], take["matched_wav"]))
if not paths:
    raise ValueError("audition manifest contains no audio")
for relative in paths:
    audio, rate = sf.read(root / relative, dtype="float32", always_2d=True)
    if rate != 44100 or audio.shape[0] == 0 or audio.shape[1] != 1:
        raise ValueError(f"invalid audition audio: {relative}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audition audio: {relative}")
report = {
    "schema": 1,
    "all_finite": True,
    "sample_rate": 44100,
    "wav_references": len(paths),
    "config_sha256": digest(sys.argv[2]),
    "checkpoint_sha256": checkpoint_sha256,
}
(root / "verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, sort_keys=True))
PY

run_timed_gpu_container \
  python -m midibrave.zrave_flow_gate \
    --manifest "$partial_container/manifest.json" \
    --output "$partial_container/gate.json" \
    --fail-on-reject

mv -- "$partial" "$output"
printf 'audition=%s\n' "$output"
