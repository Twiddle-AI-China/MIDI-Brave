#!/usr/bin/env bash
# Convenience wrapper. The real entry point is cross-platform Python:
#
#   python -m midibrave.atlas_flow_local --port 18796
#
# which is what Windows and the packaged desktop app use. This only exists so
# the old command still works, and so bash users get the repo's own interpreter
# picked for them.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${ATLAS_LOCAL_PYTHON:-$ROOT/.venv-local/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=$(command -v python3 || command -v python)
[[ -n "$PYTHON" ]] || { echo "no python found; see the README" >&2; exit 72; }

exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m midibrave.atlas_flow_local "$@"
