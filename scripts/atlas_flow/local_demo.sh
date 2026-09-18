#!/usr/bin/env bash
# Run the whole Atlas Flow demo on this laptop: no GPU, no server, no tunnel.
#
#   bash scripts/atlas_flow/local_demo.sh
#   open http://127.0.0.1:18796
#
# Needs two things next to the repo, both produced by scripts/atlas_flow/fetch_local_model.sh:
#   local-model/atlas-flow-pad-v1-weights.pt   inference weights, 85 MiB
#   local-model/pad-top50-atlas.npz            the legal-region atlas, 24 KB
# and optionally local-model/evaluation/ for the held-out four-mode comparison.
#
# The runtime generates a five-second trajectory per plan and then plays it back
# with time-warping, so CPU cost lands on changes, not on sustain. Measured on
# an M2 the plan is well inside what hovering tolerates; see --threads if you
# want to hand it more or fewer cores.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${ATLAS_LOCAL_MODEL:-$ROOT/local-model}
PYTHON=${ATLAS_LOCAL_PYTHON:-$ROOT/.venv-local/bin/python}
PORT=${ATLAS_LOCAL_PORT:-18796}
# Metal roughly halves the plan on an M2 (283 ms against 567 ms, almost all of
# it in the decoder), so prefer it and fall back to CPU where it is absent.
if [[ -z "${ATLAS_LOCAL_DEVICE:-}" ]]; then
  DEVICE=$("$PYTHON" -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null || echo cpu)
else
  DEVICE=$ATLAS_LOCAL_DEVICE
fi
THREADS=${ATLAS_LOCAL_THREADS:-0}
CACHE=${ATLAS_LOCAL_CACHE:-$MODEL/cache}
WEIGHTS=$MODEL/atlas-flow-pad-v1-weights.pt
ATLAS=$MODEL/pad-top50-atlas.npz

[[ -s "$WEIGHTS" ]] || { echo "missing weights: $WEIGHTS  (run fetch_local_model.sh)" >&2; exit 72; }
[[ -s "$ATLAS" ]] || { echo "missing atlas: $ATLAS  (run fetch_local_model.sh)" >&2; exit 72; }
[[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON — create one with:
  python3 -m venv $ROOT/.venv-local
  SSL_CERT_FILE=/etc/ssl/cert.pem $ROOT/.venv-local/bin/pip install torch soundfile numpy scipy aiohttp" >&2; exit 72; }

mkdir -p "$CACHE/takes" "$CACHE/audition"

# The config only needs a valid atlas path for inference; the data paths are
# required fields that the runtime never reads, so they point at this machine.
CONFIG=$CACHE/local_pad_v1.yaml
sed -e "s#^  atlas_path:.*#  atlas_path: $ATLAS#" \
    -e "s#^  manifest:.*#  manifest: $MODEL/unused-manifest.jsonl#" \
    -e "s#^  audio_root:.*#  audio_root: $MODEL/unused-audio#" \
    -e "s#^  feature_cache:.*#  feature_cache: $CACHE/features#" \
    -e "s#^  trajectory_cache:.*#  trajectory_cache: $CACHE/trajectories#" \
    -e "s#^  output_root:.*#  output_root: $CACHE/runs#" \
    "$ROOT/configs/atlas_flow/kraken_pad_v1.yaml" > "$CONFIG"

evaluation=$MODEL/evaluation
[[ -d "$evaluation" ]] || evaluation=$CACHE/no-evaluation

echo "atlas flow, local: device=$DEVICE  threads=${THREADS:-auto}  http://127.0.0.1:$PORT"
exec env PYTHONPATH="$ROOT/src" "$PYTHON" -m midibrave.atlas_flow_demo_server \
  --config "$CONFIG" \
  --checkpoint "$WEIGHTS" \
  --cache-root "$CACHE" \
  --evaluation-root "$evaluation" \
  --web-root "$ROOT/atlas-flow-web-demo" \
  --profile local \
  --device "$DEVICE" \
  --threads "$THREADS" \
  --host 127.0.0.1 \
  --port "$PORT"
