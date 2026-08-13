#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$script_dir/common.sh"

if (( $# > 1 )); then
  printf 'usage: %s [TAXONOMY_RELATIVE]\n' "$0" >&2
  exit 2
fi

taxonomy_relative=${1:-taxonomies/serum128-audio-v1}
validate_relative_path "$taxonomy_relative"
[[ "$taxonomy_relative" == taxonomies/serum128-audio-v1 ]] ||
  die "imported audio taxonomy must be taxonomies/serum128-audio-v1"
assert_persistent_work_root
assert_single_gpu_allocation
assert_clean_checkout

pack_host="$LV_WORK_ROOT/$LV_PACK_RELATIVE"
taxonomy_host="$LV_WORK_ROOT/$taxonomy_relative"
validator="$script_dir/validate_imported_audio_taxonomy.py"
require_directory "$pack_host"
require_file "$pack_host/index.json"
require_file "$pack_host/sequences.jsonl"
require_file "$validator"

if [[ ! -d "$taxonomy_host" ]]; then
  die "imported Octopus audio taxonomy is missing: sync $taxonomy_relative before training"
fi

python3 "$validator" \
  --pack-root "$pack_host" \
  --taxonomy-root "$taxonomy_host"
printf 'imported_audio_taxonomy=%s\n' "$taxonomy_host"
