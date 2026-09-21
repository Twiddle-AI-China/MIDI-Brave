#!/usr/bin/env bash
# Expose the Kraken-hosted demo on the intranet, to named addresses only.
#
#   bash kraken_share.sh 10.88.40.57            # one colleague
#   bash kraken_share.sh 10.88.40.0/24          # the office subnet
#   bash kraken_share.sh stop
#
# Why a forwarder: kraken-gpu-run puts the container on Docker's default bridge
# with no published port, so the service is only reachable at its bridge address
# from the host itself. This binds the host's intranet address and relays.
#
# The allowlist is enforced by socat's own `range=` option, so no root and no
# firewall change is involved, and revoking access is killing this process.
# Note there is NO authentication behind the allowlist — source address is the
# only gate, which is why the default is a single host, not a subnet.
set -Eeuo pipefail

ROOT=${ATLAS_DEMO_ROOT:-/data/rolf/atlas-flow-demo}
BIND=${ATLAS_SHARE_BIND:-10.88.40.81}
PORT=${ATLAS_SHARE_PORT:-18790}
PIDFILE=$ROOT/runtime/share.pid

stop() {
  if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"
    echo "stopped sharing (pid $(cat "$PIDFILE"))"
  else
    pkill -f "TCP-LISTEN:${PORT}," 2>/dev/null && echo "stopped stray forwarder" || echo "nothing was sharing"
  fi
  rm -f "$PIDFILE"
}

[[ "${1:-}" == "stop" ]] && { stop; exit 0; }

#   bash kraken_share.sh intranet             # anyone already on the internal network
allow=${1:?"give an address or CIDR to allow, 'intranet', or 'stop'"}
[[ -s "$ROOT/runtime/endpoint.json" ]] || { echo "no endpoint published — is the job running?" >&2; exit 72; }
address=$(sed -n 's/.*"address":"\([^"]*\)".*/\1/p' "$ROOT/runtime/endpoint.json")
target=$(sed -n 's/.*"port":\([0-9]*\).*/\1/p' "$ROOT/runtime/endpoint.json")
[[ -n "$address" ]] || { echo "endpoint.json has no address" >&2; exit 72; }

# "intranet" drops the source gate. That is bounded by the bind address: this
# port lives on a private interface with no public route in (verified — the
# public :18790 gateway forwards somewhere else entirely), so the reachable set
# is exactly "hosts already on the internal network".
if [[ "$allow" == intranet ]]; then
  gate=""
  allow="any internal source"
else
  [[ "$allow" == */* ]] || allow="$allow/32"
  gate=",range=${allow}"
fi

stop >/dev/null 2>&1 || true
mkdir -p "$ROOT/runtime"
# -d -d logs the peer address of every accepted and dropped connection, which is
# how you discover what source address a colleague actually arrives with.
nohup socat -d -d \
  "TCP-LISTEN:${PORT},bind=${BIND},fork,reuseaddr${gate}" \
  "TCP:${address}:${target}" \
  >> "$ROOT/logs/share.log" 2>&1 &
echo $! > "$PIDFILE"
sleep 1
kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "forwarder failed to start; see $ROOT/logs/share.log" >&2; exit 1; }
echo "sharing http://${BIND}:${PORT}  ->  ${address}:${target}"
echo "allowed: ${allow}   (everything else is refused)"
echo "stop with: bash \$0 stop"
