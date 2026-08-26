#!/usr/bin/env bash
set -Eeuo pipefail

readonly ENV_FILE=${ATLAS_FLOW_ENV_FILE:-/etc/midibrave/atlas-flow-spark.env}
if [[ -r "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

readonly IMAGE=${ATLAS_FLOW_IMAGE:-midibrave:atlas-flow-spark-gb10-v3}
readonly ROOT=${ATLAS_FLOW_ROOT:-/data/atlas-flow-pad-v1}
readonly PORT=${ATLAS_FLOW_PORT:-18790}
readonly RUNTIME_PORT=${ATLAS_FLOW_RUNTIME_PORT:-8791}
readonly EVALUATION=$ROOT/evaluation/pad-v1

[[ -s "$EVALUATION/evaluation.json" ]] || { echo "evaluation report is missing" >&2; exit 72; }
[[ "$PORT" =~ ^[0-9]+$ && "$RUNTIME_PORT" =~ ^[0-9]+$ ]] || {
  echo "portal and runtime ports must be numeric" >&2
  exit 64
}

exec docker run --rm \
  --name midibrave-atlas-flow-spark-portal-v5 \
  --network host \
  --cpus 2 \
  --memory 1g \
  --memory-swap 1g \
  --pids-limit 384 \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$EVALUATION",dst=/evaluation,readonly \
  "$IMAGE" python -m midibrave.atlas_flow_portal \
    --evaluation-root /evaluation \
    --web-root /opt/midibrave/atlas-flow-live-dashboard \
    --runtime-url "http://127.0.0.1:${RUNTIME_PORT}" \
    --host 0.0.0.0 \
    --port "$PORT"
