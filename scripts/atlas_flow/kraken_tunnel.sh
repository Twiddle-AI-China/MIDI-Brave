#!/usr/bin/env bash
# Keep the Kraken-hosted demo reachable at http://127.0.0.1:18795 from a laptop.
#
#   bash scripts/atlas_flow/kraken_tunnel.sh &
#
# Kraken's GPU containers sit on Docker's default bridge with no published
# port, so the forward targets the container's bridge address, which the job
# writes to runtime/endpoint.json. The address changes every run, so it is
# re-read on every reconnect.
set -u

PORT=${ATLAS_DEMO_PORT:-18795}
HOST=${ATLAS_DEMO_SSH_HOST:-kraken}
ROOT=${ATLAS_DEMO_ROOT:-/data/rolf/atlas-flow-demo}

while true; do
  endpoint=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
    "cat $ROOT/runtime/endpoint.json 2>/dev/null" 2>/dev/null)
  address=$(printf '%s' "$endpoint" | sed -n 's/.*"address":"\([^"]*\)".*/\1/p')
  remote_port=$(printf '%s' "$endpoint" | sed -n 's/.*"port":\([0-9]*\).*/\1/p')
  if [[ -z "$address" ]]; then
    echo "$(date '+%F %T') no endpoint published yet — is the job running?" >&2
    sleep 15
    continue
  fi
  echo "$(date '+%F %T') forwarding 127.0.0.1:${PORT} -> ${address}:${remote_port:-$PORT}"
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=2 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=20 \
    -L "${PORT}:${address}:${remote_port:-$PORT}" "$HOST"
  echo "$(date '+%F %T') tunnel dropped (exit $?), retrying in 5s" >&2
  sleep 5
done
