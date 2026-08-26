#!/bin/bash
set -euo pipefail
umask 027

CONTRACT=/data/atlas-flow-pad-v1/contracts/octopus-v100
CHAIN=/data/projects/latent-cosmos-synth/atlas-flow-stage1-finish-20260824/MidiBrave-v2/scripts/atlas_flow/octopus_post_stage1_chain_gpu8.sbatch
ACTIVE_JOB=$CONTRACT/post-stage1-active-job-id.txt
ATTEMPTS=$CONTRACT/post-stage1-chain-submit-attempts.txt
COMPLETE=$CONTRACT/post-stage1-chain.complete
PAUSE=$CONTRACT/post-stage1-pause-requested
EXHAUSTED=$CONTRACT/post-stage1-chain-watchdog.exhausted
LOG=$CONTRACT/post-stage1-chain-watchdog.jsonl
LOCK=$CONTRACT/post-stage1-chain-watchdog.lock
REVISION=$CONTRACT/post-stage1-chain-revision.txt
PROGRESS=$CONTRACT/post-stage1-chain-progress.txt
MAX_ATTEMPTS=4

mkdir -p "$CONTRACT"
exec 9>"$LOCK"
flock -n 9 || exit 0

record() {
  local event=$1
  local detail=${2:-}
  printf '{"time":"%s","event":"%s","detail":"%s"}\n' \
    "$(date --iso-8601=seconds)" "$event" "$detail" >> "$LOG"
}

if [[ -e "$COMPLETE" ]]; then
  rm -f "$EXHAUSTED"
  record complete
  exit 0
fi

if [[ -e "$PAUSE" ]]; then
  record paused
  exit 0
fi

if ! queue=$(squeue -h -u "${USER:?USER is required}" -n atlas-post-s1 -o '%A %T'); then
  record queue_error
  exit 1
fi
if [[ -n "$queue" ]]; then
  record active "$queue"
  exit 0
fi

attempts=0
if [[ -s "$ATTEMPTS" ]]; then
  read -r attempts < "$ATTEMPTS"
fi
[[ "$attempts" =~ ^[0-9]+$ ]] || attempts=0

chain_revision=$(sha256sum "$CHAIN" | awk '{print $1}')
previous_revision=''
[[ -s "$REVISION" ]] && read -r previous_revision < "$REVISION"
if [[ "$chain_revision" != "$previous_revision" ]]; then
  attempts=0
  printf '%s\n' "$chain_revision" > "$REVISION.tmp"
  mv "$REVISION.tmp" "$REVISION"
  printf '0\n' > "$ATTEMPTS.tmp"
  mv "$ATTEMPTS.tmp" "$ATTEMPTS"
  rm -f "$EXHAUSTED"
  record revision_changed "$chain_revision"
fi

current_progress=$(python3 - "$CONTRACT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for stage in ("joint", "flow"):
    path = root / f"{stage}-recovery-state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    update = value.get("update")
    if isinstance(update, int):
        print(f"{stage}:{update}:{value.get('recovery_profile', 'unknown')}")
        break
PY
)
previous_progress=''
[[ -s "$PROGRESS" ]] && read -r previous_progress < "$PROGRESS"
if [[ -n "$current_progress" && "$current_progress" != "$previous_progress" ]]; then
  attempts=0
  printf '%s\n' "$current_progress" > "$PROGRESS.tmp"
  mv "$PROGRESS.tmp" "$PROGRESS"
  printf '0\n' > "$ATTEMPTS.tmp"
  mv "$ATTEMPTS.tmp" "$ATTEMPTS"
  rm -f "$EXHAUSTED"
  record progress "$current_progress"
fi

if (( attempts >= MAX_ATTEMPTS )); then
  touch "$EXHAUSTED"
  record exhausted "attempts=$attempts"
  exit 0
fi

bash -n "$CHAIN"
curl -fsS http://127.0.0.1:6006/data/logdir >/dev/null
job_id=$(sbatch --parsable "$CHAIN")
attempts=$((attempts + 1))
printf '%s\n' "$job_id" > "$ACTIVE_JOB.tmp"
mv "$ACTIVE_JOB.tmp" "$ACTIVE_JOB"
printf '%s\n' "$attempts" > "$ATTEMPTS.tmp"
mv "$ATTEMPTS.tmp" "$ATTEMPTS"
rm -f "$EXHAUSTED"
record submitted "job_id=$job_id attempts=$attempts"
