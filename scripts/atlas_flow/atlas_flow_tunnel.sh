#!/usr/bin/env bash
# Keep the Atlas Flow demo reachable at http://127.0.0.1:18795 from a laptop.
#
#   bash scripts/atlas_flow/atlas_flow_tunnel.sh &
#
# A plain `ssh -L` dies on sleep, a network change, or a dropped natapp tunnel,
# and the browser then just says "refused to connect". This reconnects instead.
set -u

PORT=${ATLAS_DEMO_PORT:-18795}
HOST=${ATLAS_DEMO_SSH_HOST:-spark}

while true; do
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=2 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=15 \
    -L "${PORT}:127.0.0.1:${PORT}" "$HOST"
  echo "$(date '+%F %T') tunnel dropped (exit $?), retrying in 5s" >&2
  sleep 5
done
