#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# > 1 )); then
  printf 'usage: %s [QUEUE_TSV]\n' "$0" >&2
  exit 2
fi

queue=${1:-$script_dir/tiny_categories_v1.queue.tsv}
require_file "$queue"
require_command python3
queue=$(cd -- "$(dirname -- "$queue")" && pwd)/$(basename -- "$queue")
[[ "$queue" == "$script_dir/"* ]] ||
  die "Spark monitor refuses a queue outside scripts/spark: $queue"
queue_name=$(basename -- "$queue")
queue_name=${queue_name%.tsv}
[[ "$queue_name" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "unsafe queue basename: $queue_name"
status_root="$SP_STATE_ROOT/monitor"
mkdir -p "$status_root"

python3 "$SP_CODE_ROOT/scripts/lvzihao/monitor_matrix.py" \
  --queue "$queue" \
  --work-root "$SP_WORK_ROOT" \
  --state-root "$SP_STATE_ROOT" \
  --json-output "$status_root/$queue_name.json" \
  --markdown-output "$status_root/$queue_name.md"
