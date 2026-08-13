#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# != 1 )); then
  printf 'usage: %s QUEUE_TSV\n' "$0" >&2
  exit 2
fi

queue=$1
require_file "$queue"
require_command python3
queue_name=$(basename -- "$queue")
queue_name=${queue_name%.tsv}
[[ "$queue_name" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "unsafe queue basename: $queue_name"
status_root="$LV_STATE_ROOT/monitor"
mkdir -p "$status_root"

python3 "$script_dir/monitor_matrix.py" \
  --queue "$queue" \
  --work-root "$LV_WORK_ROOT" \
  --json-output "$status_root/$queue_name.json" \
  --markdown-output "$status_root/$queue_name.md"
