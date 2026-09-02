#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( $# > 1 )); then
  printf 'usage: %s [ALLOCATION_COUNT]\n' "$0" >&2
  exit 2
fi

queue="$script_dir/tiny_categories_v1.queue.tsv"
allocations=${1:-${SP_ALLOCATIONS:-8}}

[[ -z "${SP_INITIALIZE_FROM:-}" ]] || {
  printf 'spark: pure tiny category models must train from scratch; unset SP_INITIALIZE_FROM\n' >&2
  exit 2
}
[[ -z "${LV_INITIALIZE_FROM:-}" ]] || {
  printf 'spark: refusing inherited lvzihao initializer state\n' >&2
  exit 2
}
python3 "$script_dir/generate_category_matrix.py" --check
python3 "$script_dir/validate_category_matrix.py" --queue "$queue"
exec "$script_dir/submit_queue.sh" "$queue" "$allocations"
