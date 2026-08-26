#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

readonly ENV_FILE=${ATLAS_FLOW_ENV_FILE:-/etc/midibrave/atlas-flow-spark.env}
if [[ -r "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

readonly ROOT=${ATLAS_FLOW_ROOT:-/data/atlas-flow-pad-v1}
readonly IMAGE=${ATLAS_FLOW_IMAGE:-midibrave:atlas-flow-spark-gb10-v3}
readonly PROJECT_DIR=${ATLAS_FLOW_PROJECT_DIR:-/data/atlas-flow-pad-v1/source/atlas-flow-v5/MidiBrave-v2}
readonly RUNTIME_PORT=${ATLAS_FLOW_RUNTIME_PORT:-8791}
readonly CONTRACT=$ROOT/contracts/spark-gb10/v5
readonly QUALIFICATION=$CONTRACT/qualification.complete
readonly QUAL_SCRIPT=$PROJECT_DIR/scripts/atlas_flow/spark_live_qualification_gpu1_v5.sbatch
readonly LIVE_SCRIPT=$PROJECT_DIR/scripts/atlas_flow/spark_live_runtime_gpu1_v5.sbatch
readonly ATTEMPTS=$CONTRACT/watchdog-attempts

mkdir -p "$CONTRACT"
exec 9>"$CONTRACT/watchdog.lock"
flock -n 9 || exit 0
[[ -e "$CONTRACT/pause-requested" ]] && exit 0

if curl --fail --silent --max-time 2 \
  "http://127.0.0.1:${RUNTIME_PORT}/api/health" >/dev/null; then
  printf '0\n' > "$ATTEMPTS.tmp"
  mv "$ATTEMPTS.tmp" "$ATTEMPTS"
  rm -f "$CONTRACT/watchdog.exhausted"
  exit 0
fi

if squeue -h -n atlas-flow-qual-v5,atlas-flow-live-v5 -o '%i' | grep -q .; then
  exit 0
fi

attempts=0
[[ -s "$ATTEMPTS" ]] && read -r attempts < "$ATTEMPTS"
if (( attempts >= 3 )); then
  printf '{"state":"exhausted","attempts":%d}\n' "$attempts" \
    > "$CONTRACT/watchdog.exhausted"
  exit 1
fi

if [[ -s "$QUALIFICATION" ]]; then
  script=$LIVE_SCRIPT
  kind=runtime
else
  script=$QUAL_SCRIPT
  kind=qualification
fi
[[ -s "$script" ]] || { echo "missing $kind script: $script" >&2; exit 72; }

attempts=$((attempts + 1))
printf '%d\n' "$attempts" > "$ATTEMPTS.tmp"
mv "$ATTEMPTS.tmp" "$ATTEMPTS"
job_id=$(sbatch --parsable \
  --export=ALL,ATLAS_FLOW_IMAGE="$IMAGE",ATLAS_FLOW_ROOT="$ROOT",ATLAS_FLOW_RUNTIME_PORT="$RUNTIME_PORT" \
  "$script")
printf '{"state":"submitted","kind":"%s","attempt":%d,"job_id":"%s"}\n' \
  "$kind" "$attempts" "$job_id" >> "$CONTRACT/watchdog.jsonl"
