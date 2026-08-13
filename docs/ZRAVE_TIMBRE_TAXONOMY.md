# Z-RAVE timbre proxy taxonomy sidecars

`midibrave-zrave-timbre-taxonomy` is a pure-CPU, offline tool. It groups all
retained renders by `canonical_preset_id`, computes relative proxy scores, and
writes preset allowlists for later small-model experiments. It does not change
the pack. The trainer and sampler can consume the resulting preset allowlist.

These buckets are engineering proxies, not perceptual ground-truth timbre
labels. In particular, a RAVE latent channel has no physical frequency-axis
meaning. The default latent mode therefore never calls a temporal latent score
"dark", "bright", "clean", or "noisy".

## Feature modes

| Source | Low bucket | High bucket | Measured proxy |
| --- | --- | --- | --- |
| latent | `latent_slow_proxy` | `latent_fast_proxy` | temporal frequency centroid of the centered latent trajectory |
| latent | `latent_soft_onset_proxy` | `latent_hard_onset_proxy` | early latent-norm rise relative to median norm |
| latent | `latent_static_proxy` | `latent_moving_proxy` | median frame-to-frame latent distance relative to median norm |
| latent | `latent_smooth_proxy` | `latent_irregular_proxy` | temporal spectral flatness of the latent trajectory |
| audio | `audio_dark_proxy` | `audio_bright_proxy` | normalized audio spectral centroid |
| audio | `audio_soft_attack_proxy` | `audio_hard_attack_proxy` | early RMS rise relative to median RMS |
| audio | `audio_static_proxy` | `audio_moving_proxy` | framewise RMS and spectral-centroid change |
| audio | `audio_clean_proxy` | `audio_noisy_proxy` | audio spectral flatness |

Audio buckets are still relative proxies. They should be auditioned before
being used as product labels.

## Input contract

Latent mode reads `<pack>/sequences.jsonl`, `<pack>/index.json`, and every
referenced latent shard. The index record count, exact shard set, and all shard
SHA-256 values must match. Each sequence row must contain the packed
ordering/location fields, source and category names/codes, split name/code,
canonical preset and sample IDs, MIDI note/velocity, maximum-future and
active/total lengths. These values are checked row-by-row against the
corresponding hashed shard metadata.

Audio mode additionally requires a JSONL manifest with `sample_id` and
`audio_path`. Relative audio paths are resolved from the manifest directory.
The report binds the selected sample IDs to an aggregate SHA-256 of their audio
contents, rather than trusting path strings alone.

A canonical preset is rejected if its retained renders cross train,
validation, or test splits, or if its category is inconsistent. Scores for
multiple renders of one preset are aggregated with the median.

## Usage

Default latent mode:

```bash
midibrave-zrave-timbre-taxonomy \
  --pack-root /data/serum128-pack \
  --output-root /data/taxonomy/serum128-latent-v1
```

Optional audio mode:

```bash
midibrave-zrave-timbre-taxonomy \
  --pack-root /data/serum128-pack \
  --feature-source audio \
  --manifest /data/serum128-manifest.jsonl \
  --output-root /data/taxonomy/serum128-audio-v1
```

Thresholds are fit only from train presets. Validation and test scores are
classified with those frozen thresholds; they never influence the threshold.
The default is the train median (`--threshold-quantile 0.5`). Every resulting
bucket must satisfy the configured preset and record floors, including
per-split preset floors. A too-small bucket aborts the build before the final
output directory is published.

## Artifacts

The output directory contains:

- `taxonomy.report.json`: feature version, disclaimer, train-only quantile
  thresholds, limits, input hashes, and preset/record counts by split and
  category.
- `presets.jsonl`: one row per canonical preset with aggregated proxy scores,
  assigned buckets, split, category, source names, record count, and sample IDs.
- `buckets/<bucket>.json`: versioned bucket metadata and all preset IDs.
- `buckets/<bucket>.jsonl`: one versioned allowlist row per preset. The stable
  identifier field is `canonical_preset_id`.
- `buckets/<bucket>.ids.txt`: one canonical preset ID per line.
- `buckets/<bucket>.<split>.ids.txt`: split-specific preset IDs for train,
  validation, and test.

For training, prefer the bucket JSON because it carries `source_hashes` for the
pack index, `sequences.jsonl`, and latent shards. The JSONL and text ID files are
convenient for inspection or deliberate manual editing, but have weaker
provenance and should not be treated as proof that the allowlist belongs to the
current pack. The trainer hashes the complete allowlist file into
`data_selection_sha256`, so resume and same-profile initialization cannot
silently switch buckets. It does not yet compare a bucket JSON's embedded
`source_hashes` with the current pack; that provenance check remains a required
preflight before promoting a color-proxy model.

## lvzihao continuous-training preflight

`scripts/lvzihao/taxonomy.sh` performs that missing external provenance
preflight for the continuous queue. It defaults to the host directory
`$LV_WORK_ROOT/taxonomies/serum128-latent-v1`, runs the CPU builder inside the
pinned Docker/qgpu allocation when the directory is absent, and publishes only
through the builder's atomic directory rename. On every invocation it checks
that `taxonomy.report.json` and all eight bucket JSON files declare
`feature_source=latent` and carry the SHA-256 of the current
`$LV_WORK_ROOT/$LV_PACK_RELATIVE/index.json` and `sequences.jsonl`. It also
requires every bucket to repeat the report's sequences hash and rejects missing
or extra bucket JSONs. A valid directory is reused idempotently; a stale or
partial directory is never overwritten automatically.

The generated `tiny_latent_proxies_v1.queue.tsv` runs this preflight before
training any proxy specialist. Its configs use the bucket JSON files directly
as `preset_allowlist`, restricted to the formal Arp/Bass/FX/Lead/Pad/Pluck/Synth
categories. The queue retains 1k/5k/10k weights but deliberately postpones
all 24 audio reports until every specialist finishes training. Each report's
audition selector filters reference presets by the same allowlist, verifies
that selected preset IDs remain in that bucket, and records the allowlist hash
in its manifest.

## Imported audio-taxonomy contract

The RTX 5080 host does not carry the complete authoritative Serum WAV corpus,
so it must never create `serum128-audio-v1` locally. Build and seal it on
Octopus, where `/data/datasets/Timbre_A/serum-octopus-v1` and the matching
Serum128 pack are both available. From an Octopus SLURM allocation, with the
output directory absent, run exactly:

The checked-in, idempotent wrapper
`scripts/cloud/octopus_serum128_audio_taxonomy.sbatch` is the preferred
executable path. It requests one V100, 12 CPUs and 15 GiB, runs Docker with the
exact allocated GPU, parallelizes audio read/hash work across those CPUs,
writes only below the submitting user's artifact directory, then seals or
revalidates the output. The expanded commands below document the same contract
for an administrator-owned `/data` destination.

```bash
mkdir -p /data/midibrave-zrave-flow-serum128/taxonomies
srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm --network none \
  -v /home/yfhuang/Latent-Cosmos-Synth-midiBrave:/workspace/Latent-Cosmos-Synth-midiBrave:ro \
  -v /data/midibrave-zrave-flow-serum128:/data/midibrave-zrave-flow-serum128 \
  -v /data/datasets/Timbre_A/serum-octopus-v1:/source:ro \
  -e PYTHONPATH=/workspace/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2/src \
  -w /workspace/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2 \
  midibrave:v0.10.0-zrave-flow \
  python -m midibrave.zrave_timbre_taxonomy \
  --pack-root /data/midibrave-zrave-flow-serum128/packs/serum-balanced \
  --feature-source audio \
  --manifest /data/midibrave-zrave-flow-serum128/manifests/serum-balanced.jsonl \
  --output-root /data/midibrave-zrave-flow-serum128/taxonomies/serum128-audio-v1
```

Then seal that immutable export while the manifest and raw WAVs are still
mounted. The seal command rehashes the manifest and every selected sample's
audio content; it creates `octopus-export.provenance.json` once and refuses to
overwrite an existing receipt:

```bash
srun --kill-on-bad-exit=1 --cpu-bind=cores \
  docker run --rm --network none \
  -v /home/yfhuang/Latent-Cosmos-Synth-midiBrave:/workspace/Latent-Cosmos-Synth-midiBrave:ro \
  -v /data/midibrave-zrave-flow-serum128:/data/midibrave-zrave-flow-serum128 \
  -v /data/datasets/Timbre_A/serum-octopus-v1:/source:ro \
  -e PYTHONPATH=/workspace/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2/src \
  -w /workspace/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2 \
  midibrave:v0.10.0-zrave-flow \
  python scripts/lvzihao/validate_imported_audio_taxonomy.py \
  --pack-root /data/midibrave-zrave-flow-serum128/packs/serum-balanced \
  --taxonomy-root /data/midibrave-zrave-flow-serum128/taxonomies/serum128-audio-v1 \
  --seal-octopus-export
```

Synchronize the complete sealed directory, without editing its files, to
`$HOST_FLOW_ROOT/taxonomies/serum128-audio-v1` on lvzihao. The
`taxonomy_validate_audio` queue action performs validation only. It requires
the exact destination name, current pack index and sequences hashes, audio
feature source, exactly eight audio bucket JSONs, the complete shard map,
manifest SHA, sample/audio-content collection SHA, and the Octopus export
receipt. Every bucket must repeat the report's exact source provenance. A
missing, stale, partial, locally manufactured, or already modified directory
blocks training; the action never invokes the builder and never overwrites it.

The names `audio_dark_proxy`, `audio_bright_proxy`, `audio_clean_proxy`, and
the other audio buckets remain train-relative engineering proxies rather than
product timbre labels. `audio_static_proxy`/`audio_moving_proxy` describe
framewise RMS and centroid change, not literal note or event density.
`audio_soft_attack_proxy`/`audio_hard_attack_proxy` measure an early-window RMS
rise and are not a complete, note-off-aware ADSR measurement.
