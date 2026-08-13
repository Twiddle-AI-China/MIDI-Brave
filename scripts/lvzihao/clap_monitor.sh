#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 2 )); then
  printf 'usage: %s REPORT_ID AUDITION_ID\n' "$0" >&2
  exit 2
fi

report_id=$1
audition_id=$2
[[ "$report_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid CLAP report id: $report_id"
[[ "$audition_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid audition id: $audition_id"
assert_persistent_work_root
assert_single_gpu_allocation
assert_clean_checkout

: "${LV_CLAP_CHECKPOINT:?LV_CLAP_CHECKPOINT is required}"
: "${LV_CLAP_EXPECTED_SHA256:=fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd}"
: "${LV_CLAP_CONFIG_RELATIVE:=configs/zrave/clap_audio_monitor_v1.json}"
: "${LV_CLAP_CACHE_RELATIVE:=clap-monitor-cache/v1}"
[[ "$LV_CLAP_EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "LV_CLAP_EXPECTED_SHA256 must be a lowercase SHA-256"
validate_relative_path "$LV_CLAP_CONFIG_RELATIVE"
validate_relative_path "$LV_CLAP_CACHE_RELATIVE"
[[ "$LV_CLAP_CACHE_RELATIVE" == clap-monitor-cache/* ]] ||
  die "LV_CLAP_CACHE_RELATIVE must be below clap-monitor-cache/"
require_file "$LV_CLAP_CHECKPOINT"
[[ "$LV_CLAP_CHECKPOINT" == "$LV_WORK_ROOT"/* ]] ||
  die "LV_CLAP_CHECKPOINT must be below LV_WORK_ROOT"
checkpoint_name=$(basename -- "$LV_CLAP_CHECKPOINT")
[[ "$checkpoint_name" == music_audioset_epoch_15_esc_90.14.pt ]] ||
  die "unsupported CLAP checkpoint filename: $checkpoint_name"
checkpoint_bytes=$(wc -c <"$LV_CLAP_CHECKPOINT" | tr -d '[:space:]')
[[ "$checkpoint_bytes" == 2352471003 ]] ||
  die "CLAP checkpoint byte size mismatch: $checkpoint_bytes"

checkpoint_sha=$(sha256sum "$LV_CLAP_CHECKPOINT" | awk '{print $1}')
[[ "$checkpoint_sha" == "$LV_CLAP_EXPECTED_SHA256" ]] ||
  die "CLAP checkpoint SHA-256 mismatch"
config_host=$(host_config_path "$LV_CLAP_CONFIG_RELATIVE")
config_container=$(container_config_path "$LV_CLAP_CONFIG_RELATIVE")
require_file "$config_host"
config_sha=$(sha256sum "$config_host" | awk '{print $1}')
checkpoint_container=$(host_path_to_container_work_path "$LV_CLAP_CHECKPOINT")
cache_host="$LV_WORK_ROOT/$LV_CLAP_CACHE_RELATIVE"
cache_container="$LV_CONTAINER_WORK_ROOT/$LV_CLAP_CACHE_RELATIVE"
manifest_host="$LV_WORK_ROOT/auditions/$audition_id/manifest.json"
manifest_container="$LV_CONTAINER_WORK_ROOT/auditions/$audition_id/manifest.json"
require_file "$manifest_host"
manifest_sha=$(sha256sum "$manifest_host" | awk '{print $1}')
output_root="$LV_WORK_ROOT/clap-reports"
output="$output_root/$report_id.json"
mkdir -p "$output_root" "$cache_host"

if [[ -f "$output" ]]; then
  if python3 - "$output" "$manifest_host" "$manifest_sha" \
    "$checkpoint_sha" "$config_sha" <<'PY'
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
manifest = Path(sys.argv[2]).resolve()
root = manifest.parent
contract = report.get("clap_contract", {})
rows = report.get("rows")
audio_hashes_current = isinstance(rows, list) and bool(rows)
identifiers = set()
if audio_hashes_current:
    for row in rows:
        paths = row.get("audio_paths") if isinstance(row, dict) else None
        hashes = row.get("audio_sha256") if isinstance(row, dict) else None
        identifier = row.get("id") if isinstance(row, dict) else None
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(paths, dict)
            or not isinstance(hashes, dict)
            or set(paths) != {"source", "direct", "generated"}
            or set(hashes) != {"source", "direct", "generated"}
        ):
            audio_hashes_current = False
            break
        identifiers.add(identifier)
        for role in ("source", "direct", "generated"):
            relative = paths[role]
            expected = hashes[role]
            if not isinstance(relative, str) or not isinstance(expected, str):
                audio_hashes_current = False
                break
            candidate = Path(relative)
            if candidate.is_absolute() or any(
                part in {".", ".."} for part in candidate.parts
            ):
                audio_hashes_current = False
                break
            resolved = (root / candidate).resolve()
            if (
                (resolved.parent != root and root not in resolved.parents)
                or not resolved.is_file()
                or digest(resolved) != expected
            ):
                audio_hashes_current = False
                break
        if not audio_hashes_current:
            break
valid = (
    report.get("kind") == "zrave-frozen-clap-audio-preservation-report"
    and report.get("report_only") is True
    and report.get("manifest_sha256") == sys.argv[3]
    and contract.get("checkpoint_sha256") == sys.argv[4]
    and contract.get("config_sha256") == sys.argv[5]
    and audio_hashes_current
)
raise SystemExit(0 if valid else 1)
PY
  then
    printf 'CLAP report already complete: %s\n' "$output"
    exit 0
  fi
  die "CLAP report belongs to another contract or references stale WAVs: $output"
fi

output_container="$LV_CONTAINER_WORK_ROOT/clap-reports/$report_id.json"

# This action is opt-in. The normal lvzihao image intentionally omits the CLAP
# stack; a dedicated queue must set LV_IMAGE to an audited CLAP-capable image.
run_timed_gpu_container python -c '
import importlib.metadata
import torchlibrosa
import transformers
from transformers import (
    BartTokenizer,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
)
expected = {
    "laion-clap": "1.1.7",
    "transformers": "5.13.0",
    "torchlibrosa": "0.1.0",
    "librosa": "0.11.0",
    "ftfy": "6.3.1",
}
observed = {name: importlib.metadata.version(name) for name in expected}
assert observed == expected, (observed, expected)
BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True)
RobertaModel.from_pretrained("roberta-base", local_files_only=True)
BartTokenizer.from_pretrained("facebook/bart-base", local_files_only=True)
import laion_clap
assert laion_clap is not None
assert torchlibrosa is not None
assert transformers is not None
'

run_timed_gpu_container \
  python -m midibrave.zrave_clap_monitor \
    --manifest "$manifest_container" \
    --checkpoint "$checkpoint_container" \
    --expected-checkpoint-sha256 "$LV_CLAP_EXPECTED_SHA256" \
    --config "$config_container" \
    --cache-root "$cache_container" \
    --output "$output_container" \
    --device cuda

require_file "$output"
printf 'clap_report=%s\n' "$output"
