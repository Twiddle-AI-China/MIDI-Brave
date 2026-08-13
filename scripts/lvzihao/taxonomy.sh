#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# > 1 )); then
  printf 'usage: %s [TAXONOMY_RELATIVE]\n' "$0" >&2
  exit 2
fi

taxonomy_relative=${1:-taxonomies/serum128-latent-v1}
validate_relative_path "$taxonomy_relative"
[[ "$taxonomy_relative" == taxonomies/* ]] ||
  die "taxonomy output must be below taxonomies/: $taxonomy_relative"
assert_persistent_work_root
assert_single_gpu_allocation
assert_clean_checkout

pack_host="$LV_WORK_ROOT/$LV_PACK_RELATIVE"
pack_container="$LV_CONTAINER_WORK_ROOT/$LV_PACK_RELATIVE"
taxonomy_host="$LV_WORK_ROOT/$taxonomy_relative"
taxonomy_container="$LV_CONTAINER_WORK_ROOT/$taxonomy_relative"
validator="$script_dir/validate_timbre_taxonomy.py"
require_directory "$pack_host"
require_file "$pack_host/index.json"
require_file "$validator"

validate_current_taxonomy() {
  python3 "$validator" \
    --pack-root "$pack_host" \
    --taxonomy-root "$taxonomy_host"
}

if [[ -e "$taxonomy_host" ]]; then
  [[ -d "$taxonomy_host" ]] ||
    die "taxonomy output exists but is not a directory: $taxonomy_host"
  validate_current_taxonomy
  printf 'taxonomy already complete: %s\n' "$taxonomy_host"
  exit 0
fi

mkdir -p "$(dirname -- "$taxonomy_host")"
run_timed_gpu_container \
  python -m midibrave.zrave_timbre_taxonomy \
    --pack-root "$pack_container" \
    --output-root "$taxonomy_container" \
    --feature-source latent

validate_current_taxonomy
printf 'taxonomy=%s\n' "$taxonomy_host"
