#!/usr/bin/env bash
# Pull everything a laptop needs to run Atlas Flow without the cluster.
#
#   bash scripts/atlas_flow/fetch_local_model.sh
#
# Weights are exported inference-only: the training checkpoint is 255 MB, of
# which 170 MB is Adam optimizer state that inference never touches.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${ATLAS_LOCAL_MODEL:-$ROOT/local-model}
HOST=${ATLAS_LOCAL_SSH_HOST:-kraken}
REMOTE=${ATLAS_REMOTE_ROOT:-/data/rolf/atlas-flow-demo}
DATA=${ATLAS_REMOTE_DATA:-/data/atlas-flow-pad-v1}
CHECKPOINT=$DATA/runs/pad-v1/joint/checkpoints/step-000030000.pt

mkdir -p "$MODEL"

if ! ssh -o BatchMode=yes "$HOST" "test -s $REMOTE/atlas-flow-pad-v1-weights.pt"; then
  echo "exporting inference weights on $HOST…"
  ssh -o BatchMode=yes "$HOST" "cd $REMOTE && kraken-cpu-run --docker --memory 8G --cpus 2 \
    --name atlas-export --mount $REMOTE:/work --mount $DATA:$DATA:ro \
    --mount $REMOTE/repo/src:/opt/midibrave/src:ro -- \
    midibrave:atlas-flow-kraken-v100-v1 python /work/export_weights.py \
    $CHECKPOINT /work/atlas-flow-pad-v1-weights.pt"
fi

echo "fetching weights (85 MiB)…"
rsync -az --partial --info=progress2 -e 'ssh -o BatchMode=yes' \
  "$HOST:$REMOTE/atlas-flow-pad-v1-weights.pt" "$MODEL/"
echo "fetching atlas…"
rsync -az --partial -e 'ssh -o BatchMode=yes' "$HOST:$DATA/atlas/pad-top50-atlas.npz" "$MODEL/"
echo "fetching held-out evaluation bundle (optional, 36 MB)…"
rsync -az --partial -e 'ssh -o BatchMode=yes' "$HOST:$DATA/evaluation/pad-v1/" "$MODEL/evaluation/" || \
  echo "  skipped — the four-mode comparison will be unavailable"

du -sh "$MODEL"/* 2>/dev/null
echo "now: bash scripts/atlas_flow/local_demo.sh"
