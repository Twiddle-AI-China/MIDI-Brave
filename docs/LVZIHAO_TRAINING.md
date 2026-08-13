# lvzihao / RTX 5080 training

This directory provides a reproducible single-GPU path for the Serum128
Z-RAVE baseline and the new segment-2/4/8 pure and MIDI-conditioned training
families. Existing Octopus YAML remains usable by mounting the lvzihao host
runtime at the paths already present in the YAML.

## Invariants

- Every CUDA command refuses to run without `SLURM_JOB_ID` and exactly one
  `CUDA_VISIBLE_DEVICES` entry. Docker receives that exact device; the scripts
  never use `--gpus all` and never launch GPU Python from the login shell.
- Submission is only through `qgpu`, with one GPU, at most 72 hours, at most 28
  CPU cores, and a hard 32G memory guard. The default is 32G because that is the
  measured usable qgpu ceiling on lvzihao. Docker is independently capped at
  eight CPUs and 30G RAM so the daemon cannot escape the allocation budget.
- Code is mounted read-only. Packs, checkpoints, TensorBoard data, sweep
  reports, audition WAVs, queue state, and caches persist below
  `/home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration/runtime/serum128`.
- The queue records the queue SHA-256, Git commit, Docker image ID, and mount
  contract. A dirty checkout or a mid-queue code/image change stops the queue.
- Runtime containers are offline and receive only explicit environment
  variables. Credentials are neither stored nor forwarded.

The CUDA base is the official
`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` linux/amd64 manifest pinned at
`sha256:7b324d...d51385`. PyTorch's release matrix lists CUDA 12.8 as stable for
2.9, and the CUDA 12.8 build covers Blackwell compute capability 12.0. The
smoke step additionally asserts that `sm_120` is compiled in, detects one
compute-capability-12.x GPU, executes a CUDA matrix multiplication, and runs a
short real Z-RAVE benchmark.

## Host layout

The defaults match the deployed lvzihao layout:

```text
/home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration/
├── repo/                         # clean Git checkout
│   └── MidiBrave-v2/
└── runtime/
    ├── serum128/                 # all mutable flow artifacts
    │   ├── codec/
    │   ├── packs/serum-balanced/
    │   ├── runs/
    │   ├── sweeps/
    │   ├── auditions/
    │   └── lvzihao-state/
    ├── rave-serum-v2/            # read-only corpus metadata input
    └── source/serum-octopus-v1/  # read-only source audio input
```

Before a sweep, `runtime/serum128` must contain the prepared assets expected
by the selected config, notably `packs/serum-balanced/index.json`,
`packs/serum-balanced/statistics.npz`, and the codec used for audition. The
scripts do not download data or model weights. Synchronize verified assets
separately, preserve the paths above, and record their source hashes.

## Build and smoke

On the lvzihao login host, build the image without requesting a GPU:

```bash
cd /home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration/repo/MidiBrave-v2
bash scripts/lvzihao/build_image.sh
```

If Docker Hub is unreachable but the already-audited local ACE-Step CUDA 12.8
image is present, the build script supports this explicit, digest-pinned
offline-base fallback:

```bash
LV_BASE_IMAGE=ghcr.io/ace-step/ace-step-1.5@sha256:c7391f0e1b06c723486015aa53641883cd6a46ba167cf13918d4acadc6c75d77 \
LV_EXPECTED_TORCH_PREFIX=2.10 \
bash scripts/lvzihao/build_image.sh
```

That fallback reuses only the base image's Python 3.11 / Torch 2.10 / CUDA
12.8 userspace and installs the pinned MidiBrave dependencies into the derived
image. It is not interchangeable silently: record the resulting image ID in a
new queue contract and require the qgpu smoke to prove `sm_120`, CUDA matmul,
the real pack path, and weight-only initialization before training.

Copy `scripts/lvzihao/env.example` outside the checkout, adjust only the input
paths, and source it. Do not add credentials:

```bash
cp scripts/lvzihao/env.example \
  /home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration/lvzihao.env
. /home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration/lvzihao.env
```

Both segment queues require the imported 85k pure baseline as a weight-only
initializer. Export it before previewing or submitting either queue:

```bash
export LV_INITIALIZE_FROM="$HOST_FLOW_ROOT/runs/pure-flow-v1/checkpoints/step-085000.pt"
```

Preview the exact qgpu commands without submitting:

```bash
LV_SUBMIT_DRY_RUN=1 bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/segment248_pure.queue.example.tsv 2
```

Then submit enough identical continuation workers to span the desired number
of 72-hour windows:

```bash
bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/segment248_pure.queue.example.tsv 8
```

qgpu owns the GPU allocation. Each worker runs the same serial queue under a
file lock. Extra workers exit quickly once the queue is complete.

The smoke action uses batch 5 so a 20% MIDI-transition schedule executes at
least one real within-future event per update. When `LV_INITIALIZE_FROM` is
set, smoke also performs the strict shared-backbone weight-only load before
its CUDA updates; incompatible 85k lineage therefore fails before the sweep.

`pure_flow.queue.example.tsv` is baseline-only. It can smoke, benchmark, and
audition the imported Octopus pure-flow run, but intentionally has no `train`
row. Never use a single-GPU `--resume` to turn its 85k multi-GPU checkpoint
into a 100k lvzihao run; that would violate the saved world-size/batch contract.

The segment-2/4/8 MIDI queue additionally needs the pitch-probe root. Export it
before submission. `LV_INITIALIZE_FROM` is consumed as weight-only
initialization when no segment run checkpoint exists; `LV_PITCH_PROBE` is consumed
by smoke, sweep, and train after the queue's first `pitch_train` row creates and
qualifies it:

```bash
export LV_PITCH_PROBE="$HOST_FLOW_ROOT/runs/pitch-probe"
bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/segment248_midi.queue.example.tsv 8
```

Use `pitch_probe.queue.example.tsv` instead when only the independently
resumable 128D pitch probe is wanted. Both G1 and G3 queues first finish the
entire 20k training row, which produces `step-005000.pt`, `step-010000.pt`,
`step-015000.pt`, and `step-020000.pt`. Only after all four weights exist does
the queue audition and gate them in ascending update order. Thus an early
audio rejection cannot prevent the later planned weights from being trained.

Each G3 `midi_audition` row requires `LV_SOURCE_ROOT`, `LV_PITCH_PROBE`, and
`LV_INITIALIZE_FROM`, hashes the selected flow checkpoint and initializer,
renders matched, observed-note-swap, within-future `note_step`, and
`velocity_step` audio, then applies three independent
reports: the generic 128D long-rollout degeneration gate, the qualified
latent-pitch-probe adherence gate, and decoded-audio CREPE MIDI adherence. The
latent gate remains an explicit proxy. The decoded gate separately tracks only
the generated WAV interval and reports voiced coverage, cents, octave errors,
and note-step transition settling while validating its official torchcrepe
model hash. Velocity following is reported separately as a before/after steady
RMS-direction proxy; it is not called velocity accuracy.

Build the offline evaluator before the first MIDI audition. It does not alter
the lean training image or any weight-production row:

```bash
bash scripts/lvzihao/build_audio_eval_image.sh
export LV_AUDIO_EVAL_IMAGE=midibrave:lvzihao-cu128-audio-eval-v1
```

When the trainer itself was built from the explicit Torch 2.10 ACE-Step
fallback above, pass the matching audited versions to the derived evaluator as
well:

```bash
LV_AUDIO_EVAL_EXPECTED_TORCH_PREFIX=2.10 \
LV_AUDIO_EVAL_EXPECTED_TORCHAUDIO_PREFIX=2.10 \
bash scripts/lvzihao/build_audio_eval_image.sh
```

## Queue and resume behavior

The queue is a five-column, tab-separated file:

```text
experiment_id  action  config_relative  run_relative  spec
```

Actions and `spec` values are:

- `taxonomy`: `taxonomies/<name>` (or `-` for the default
  `taxonomies/serum128-latent-v1`). It runs the CPU latent-proxy builder inside
  the pinned Docker/qgpu allocation, then requires the current pack index and
  sequences hashes, latent feature source, and exactly eight valid bucket JSON
  allowlists. A valid existing output is reused; a stale or partial output
  blocks the queue.
- `taxonomy_validate_audio`: only `taxonomies/serum128-audio-v1` (or `-` for
  that default), with `-` in config/run. It validates an immutable audio
  taxonomy built and raw-audio-sealed on Octopus. It never builds, repairs, or
  overwrites the imported directory; missing provenance or a pack/content hash
  mismatch blocks training.
- `clap_report`: an audition ID in `spec`, with `-` in the config/run fields.
  This is an independent, report-only frozen-CLAP audio/audio comparison. It is
  never present in the category, proxy, or MIDI training queues and cannot
  block weight production unless a user explicitly submits a separate CLAP
  report queue.
- `smoke`: `-`; checks Blackwell and executes a tiny data-backed benchmark.
- `pitch_train`: maximum update count. It trains the 128D latent pitch probe,
  resumes the highest `step-*.pt` after an allocation boundary, and completes
  only when `qualification.json` says `passed: true` and its checkpoint exists.
  `LV_PITCH_BATCH_PER_GPU` defaults to 64 for the 16G RTX 5080.
- `sweep`: comma-separated per-GPU batches. Completed valid reports are reused;
  OOM/failed candidates are recorded and the fastest safe single-GPU batch is
  written to `<run>/lvzihao-selection.json`.
- `train`: the fixed maximum update count. It verifies the sweep against the
  current config, pack, and Git commit, then resumes the highest `step-*.pt`.
- `midi_smoke_from`, `midi_sweep_from`, and `midi_train_from`: the corresponding
  action spec followed by `|runs/.../checkpoints/step-NNNNNN.pt`. The runner
  resolves the initializer below the persistent work root, rejects traversal
  and a global `LV_INITIALIZE_FROM`, persists its path/SHA-256/update contract,
  and exports that exact same-family initializer only for the current row.
- `audition`: `latest`, `final.pt`, or an explicit `step-N.pt` basename. Output
  is promoted from a partial directory only after every referenced WAV is
  finite and has the expected shape/rate.
- `audition_report`: the same pure-flow render and generic degeneration gate,
  but a valid `gate.json` with `passed: false` is retained and returns success
  so a research matrix can evaluate every planned weight. Render, contract,
  hash, or WAV-integrity failures still block the queue. Production promotion
  must use the hard `audition` action.
- `midi_audition`: the same checkpoint spec. It strictly validates the MIDI
  architecture, checkpoint/initializer/pitch-probe/config/data hashes, renders
  the config's categories, and promotes output only when the generic,
  latent-pitch-probe, and decoded-audio CREPE adherence gates pass.
- `midi_audition_report`: the same global-initializer render but retains valid
  negative gate reports. `midi_audition_from` and
  `midi_audition_report_from` take `CHECKPOINT|INITIALIZER_RELATIVE`; they add
  the per-row initializer contract used by the tiny MIDI matrix. Report mode
  softens only gate rejection: render, lineage/hash, pitch-probe qualification,
  manifest linkage, and WAV-integrity failures remain hard.

The allocation runner uses a 70-hour execution budget within each 72-hour
qgpu request, leaving shutdown margin. Current training writes atomic periodic
checkpoints but has no signal-triggered checkpoint. At a boundary Docker is
stopped and the next prequeued worker resumes the newest completed checkpoint;
at most one checkpoint interval is replayed. Choose a config whose
`checkpoint_every` comfortably fits the measured 70-hour window.

A hard SLURM cutoff is also recoverable: the next worker sees that the train
row has no completion marker and resumes it. Sweeps resume at the first batch
without a valid report. Auditions keep failed partial output for diagnosis and
retry into a new job-specific partial directory.

Queue state and logs live in
`runtime/serum128/lvzihao-state/queues/<queue-name>/`. A non-timeout failure
creates `blocked.txt` so queued workers do not repeatedly run a broken task.
Inspect the referenced log, fix the cause, move `blocked.txt` aside for the
audit trail, and submit continuation workers again. Do not alter a live queue,
checkout, or image tag; start a new queue filename for a changed experiment.

## Monitoring contract

Every successful log interval is written twice by rank 0: TensorBoard scalars
for interactive curves and an append-only, flushed `metrics.jsonl` for simple
host-side inspection. The JSONL contains the complete finite scalar set, not
only a dashboard subset. It covers:

- total, flow, boundary, statistics, temporal, and pitch-probe loss terms;
- learning rate, gradient norm, AMP scale, non-finite skips, and valid latent
  frames/second;
- mask-aware future lengths and two deliberately separate distributions:
  `target/*` describes sampled training data, while `estimate/*` describes the
  current flow field's one-step clean-latent estimate. Both include latent
  mean/std, frame-norm percentiles, normalized temporal motion, near-static
  and normalized-coordinate near-zero fractions, plus observed/reference
  per-channel std ratios;
- source/category coverage, segment division/index/joint proportions, and—when
  MIDI is enabled—condition dropout, transitions/events, note and velocity
  distributions plus explicitly named `probe/*` pitch diagnostics.

只有 `estimate/*` 能提示模型自身是否趋向静态或低方差吸引子；
`target/*` 是数据/控制基线。“Near zero” 和 “near static” 仍然只是
latent 分布代理，不是可听稀疏度、静音或音符密度的声明。Pure and MIDI
models both run a deterministic validation panel at the configured interval.
Its sample panel and flow noise are fixed, while its sampler/RNG/model mode are
restored so validation cannot perturb training. If a hard allocation cutoff
leaves telemetry ahead of the newest checkpoint, resume appends a
`resume_rollback_to_checkpoint` event and keeps the orphaned observations; it
does not delete or silently reinterpret them.

Use the read-only matrix monitor from the login host:

```bash
bash scripts/lvzihao/monitor_matrix.sh \
  scripts/lvzihao/tiny_categories_v1.queue.tsv
watch -n 30 bash scripts/lvzihao/monitor_matrix.sh \
  scripts/lvzihao/tiny_midi_categories_v1.queue.tsv
```

It prints a compact table and atomically writes full JSON plus Markdown below
`runtime/serum128/lvzihao-state/monitor/`. The snapshot includes every latest
scalar, checkpoint updates, queue completion/blocking, generic audio gate,
envelope-proxy summary, latent and decoded-audio MIDI gates, explicit
qualification status, and CLAP-report presence.

Each 1k/5k/10k offline audition computes the generic long-rollout health gate
(finite audio, silence, tail RMS drift, static tone, short cycles, boundaries,
and seed diversity) and report-only amplitude-envelope proxies for attack,
peak-to-sustain decay, sustain variation, and tail slope. Those envelope
values are deliberately not called true ADSR: current renders do not carry an
authoritative note-off sample, so a real Release measurement requires the data
renderer to add note-on/note-off timing and a release tail. MIDI auditions also
run the qualified latent-pitch-probe adherence gate; decoded-audio pitch is a
separate evaluator and must not be inferred from that proxy. CLAP is a separate
report-only queue until its codec ceiling and negative controls establish a
calibrated threshold.

Every fresh audition also writes `qualification.json`. A soft report is always
`research_only`, even when its metrics pass. A hard pure-flow audition can be
`qualified`. MIDI hard results are currently recorded as
`provisional_gates_passed_uncalibrated`, never production-qualified, because
the decoded-audio threshold still needs same-panel Source/RAVE-Direct ceiling
calibration. Queue completion and `final.pt` therefore do not mean a weight has
passed promotion.

## Segment-2/4/8 experiment families

The pure and MIDI templates use immutable, distinct YAML/output contracts:

- `segment248_pure.queue.example.tsv` targets
  `runs/segment248-pure`, sweeps batches `1,2,4,8,12,16`, and trains to 20k.
  It requires the global `LV_INITIALIZE_FROM` export shown above. After the
  complete train row, four `audition` rows gate its 5k/10k/15k/20k weights.
- `segment248_midi.queue.example.tsv` targets
  `runs/segment248-midi32`, first qualifies the pitch probe, sweeps batches
  `1,2,4,8`, and trains to 20k. It then runs four `midi_audition` rows for its
  5k/10k/15k/20k weights. It requires the global `LV_INITIALIZE_FROM` and
  `LV_PITCH_PROBE` exports shown above.

Both resolve their data through these host/container variables:

- `HOST_REPO_ROOT` -> `/workspace/Latent-Cosmos-Synth` (read-only)
- `HOST_FLOW_ROOT` -> `/data/midibrave-zrave-flow-serum128` (read/write)
- `LV_RAVE_ROOT` -> `/data/midibrave-rave-serum-v2` (read-only)
- `LV_SOURCE_ROOT` -> `/source` (read-only)

`LV_PACK_RELATIVE` defaults to `packs/serum-balanced`; change it together with
the YAML `data.packed_root` when a new prepared pack lands. The runner rejects
YAML output/pack paths that disagree with its persistent mount contract.

Keep every architecture/training variant on a distinct `train.output_root` and
queue filename. `LV_INITIALIZE_FROM` and `LV_PITCH_PROBE` must be absolute host
paths below `HOST_FLOW_ROOT`; the scripts translate them to container paths.
Automatic same-run resume always takes precedence over the initializer.

## Current Serum128 data inventory

The source corpus is the private Octopus dataset at
`/data/datasets/Timbre_A/serum-octopus-v1`; it is not downloadable from this
Git repository. The synchronized predictor pack contains 134,885 retained
renders (129,501 train / 2,701 validation / 2,683 test), encoded by the frozen
128D RAVE codec. Each retained Serum preset can contribute the six
note/velocity renders from notes 36/62/82 and velocities 54/108. Splits are
preset-wise, so the same canonical preset cannot leak across train and held-out
sets.

| Category | Train | Validation | Test | Total | First-matrix policy |
| --- | ---: | ---: | ---: | ---: | --- |
| Arp | 5,162 | 108 | 108 | 5,378 | native |
| Atmosphere | 288 | 6 | 6 | 300 | exclude; merge or add data |
| Bass | 26,341 | 554 | 568 | 27,463 | native |
| Chord | 1,258 | 22 | 30 | 1,310 | pilot |
| Drums | 429 | 12 | 6 | 447 | exclude; merge or add data |
| FX | 5,397 | 108 | 112 | 5,617 | native pure-flow |
| Keys | 3,685 | 78 | 84 | 3,847 | pilot and harmony role |
| Lead | 21,710 | 462 | 450 | 22,622 | native |
| Pad | 13,566 | 282 | 282 | 14,130 | native |
| Pluck | 9,434 | 188 | 191 | 9,813 | native |
| Synth | 42,098 | 875 | 846 | 43,819 | native |
| Vocal | 133 | 6 | 0 | 139 | exclude; no test coverage |

The table is an artifact inventory, not a class-balance target. Every generated
config hashes its data selection into the checkpoint contract, and every
category/preset filter is applied before latents move to the GPU. Full shards
are still read and hash-verified in host memory before compaction.

## Tiny category experiment matrix

The first category-specialist matrix is declarative:

- recipe: `scripts/lvzihao/tiny_category_recipes.yaml`;
- immutable base: `configs/zrave/lvzihao_serum128_tiny_category_template.yaml`;
- generator: `scripts/lvzihao/generate_category_matrix.py`;
- generated configs: `configs/zrave/generated/lvzihao_tiny_categories_v1/`;
- generated queue: `scripts/lvzihao/tiny_categories_v1.queue.tsv`.

Regenerate after an intentional recipe/template change, and verify that the
checked-in artifacts are current before submission:

```bash
python scripts/lvzihao/generate_category_matrix.py
python scripts/lvzihao/generate_category_matrix.py --check
```

The independent run roots below `runs/tiny-categories-v1/` contain native
families `arp`, `bass`, `fx`, `lead`, `pad`, `pluck`, `synth`, the smaller-data
pilots `chord_pilot` and `keys_pilot`, plus role combinations `lead_pluck` and
`keys_harmony`. Categories may intentionally occur in both a native family and
a role combination; each run still has its own config, checkpoints, optimizer,
RNG, sweep result, and output root. Atmosphere, Drums, and Vocal are excluded
from this first matrix because the current pack has only 300, 447, and 139
records respectively. Every family uses the exact pure `tiny` profile
`(d_model=128, context_layers=2, future_layers=4, heads=4,
feedforward_dim=512)`, segment-2/4/8 sampling, and a 10k schedule with 1k
checkpoint/validation/curriculum intervals.

The tiny sweep candidates are `16,32,64,96,128,160`, reflecting the much
smaller 1.93M-parameter predictor. The loader still reads and verifies every
pack shard in host memory, but category and preset filters compact records
before latents move to the GPU, so each family's resident data footprint is
different. These are probes, not promised-safe batch sizes: desktop or other
external GPU sidecars on the 5080 reduce available memory, OOM candidates may
fail, and the persisted finite/memory/throughput sweep reports remain the only
authority for the selected batch.

These pure tiny models train from scratch. `LV_INITIALIZE_FROM` must be unset:
standard 85k/G1 weights are shape-incompatible and the shell plus checkpoint
contracts reject that accidental initialization. A future MIDI tiny model must
instead use a qualified pure initializer with the same `tiny` profile.

The generated queue deliberately places every family's smoke, sweep, and full
10k train rows before any evaluation row. It then runs `audition_report` for
the same-trajectory `step-001000.pt`, `step-005000.pt`, and
`step-010000.pt` weights. A negative audio-health gate therefore does not hide
the remaining research results. The renderer reads the single source's ordered
`allowed_categories` from each generated config; it no longer assumes the
six-category broad-model panel. The report remains a generic degeneration
check, not MIDI adherence or a production promotion decision.

The currently synchronized lvzihao source-audio audition subset covers only
Pad, Lead, Bass, Pluck, Keys, and Synth. Before the `arp`, `chord_pilot`, `fx`,
or `keys_harmony` report rows can render all requested categories, synchronize
the matching Arp/Chord/FX source WAVs below `LV_SOURCE_ROOT`. This is an
evaluation-input prerequisite, not a training prerequisite: the generated TSV
places all eleven 10k train rows before every report row, so missing audition
source audio cannot prevent any planned model weights from being produced.

## Tiny MIDI category experiment matrix

The second stage adds sequence-conditioned specialists for the eight tonal
families with a usable MIDI contract: Arp, Bass, Chord, Keys, Lead, Pad, Pluck,
and Synth. Chord and Keys remain explicitly named pilots because of their
smaller held-out sets. Its declarative artifacts are:

- recipe: `scripts/lvzihao/tiny_midi_category_recipes.yaml`;
- immutable base:
  `configs/zrave/lvzihao_serum128_tiny_midi_category_template.yaml`;
- generated configs:
  `configs/zrave/generated/lvzihao_tiny_midi_categories_v1/`;
- training/evaluation queue:
  `scripts/lvzihao/tiny_midi_categories_v1.queue.tsv`;
- independent CLAP queue:
  `scripts/lvzihao/tiny_midi_categories_v1.clap.queue.tsv`.

Generate or validate the checked-in artifacts with:

```bash
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_midi_category_recipes.yaml
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_midi_category_recipes.yaml --check
```

Every MIDI family keeps the `tiny` architecture and segment-2/4/8 sampler,
trains for 10k updates, and writes checkpoints plus deterministic validation
at every 1k. It must initialize weight-only from exactly the same family's pure
`runs/tiny-categories-v1/<family>/checkpoints/step-010000.pt`. The generator
checks the pure config's category selection, profile/dimensions, pure flags,
run root, schedule, and checkpoint availability. The generated TSV repeats the
explicit relative initializer on every row; the allocation runner then resolves
it below `$HOST_FLOW_ROOT/runs`, records its SHA-256, and rejects any hash drift
on a later allocation. A standard 85k checkpoint or one global initializer for
all families is therefore not an accepted path through this matrix.

Use one already-qualified, globally shared pitch probe, but leave the global
initializer unset:

```bash
unset LV_INITIALIZE_FROM
export LV_PITCH_PROBE="$HOST_FLOW_ROOT/runs/pitch-probe"
bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/tiny_midi_categories_v1.queue.tsv 8
```

The queue orders all eight smoke/sweep/train trajectories before any report,
so each independent run root can resume across as many 72-hour qgpu allocations
as required and all planned weights are produced first. It then evaluates the
1k, 5k, and 10k weights with `midi_audition_report_from`. Generic degeneration,
latent-pitch-probe MIDI adherence, and decoded-audio CREPE JSON reports are
retained; negative research thresholds do not hide later weights, while
render/hash/WAV/CREPE-asset integrity failures still block. A production
candidate must be rerun through hard `midi_audition_from` and pass all three
gates.

Training monitoring is written per family to `metrics.jsonl` and TensorBoard,
including loss terms, learning rate/grad/AMP/throughput, valid-future and latent
distribution/sparsity diagnostics, segment mix, and MIDI condition/event/note/
velocity distributions. Validation is emitted every 1k. Offline MIDI reports
add matched/note-swap/note-step coverage, both latent-probe and decoded-audio
cents adherence, decoded voiced coverage, octave errors, and transition
settling. The velocity-step panel adds report-only steady RMS dB response and
requested-direction agreement. Voiced coverage, settling, and velocity response
remain report-only pending held-out calibration.
This still does not make true note-off-aware ADSR available; that requires
renderer note-on/note-off labels and a release-tail capture contract.

## Tiny latent-proxy experiment matrix

The latent-proxy specialist matrix partitions the seven formal Serum categories by
four preset-level latent-trajectory proxy axes. It is declarative and uses the
same immutable tiny/segment-2/4/8 base as the category matrix:

- recipe: `scripts/lvzihao/tiny_latent_proxy_recipes.yaml`;
- generator: `scripts/lvzihao/generate_category_matrix.py`;
- generated configs:
  `configs/zrave/generated/lvzihao_tiny_latent_proxies_v1/`;
- generated queue: `scripts/lvzihao/tiny_latent_proxies_v1.queue.tsv`;
- taxonomy output:
  `$HOST_FLOW_ROOT/taxonomies/serum128-latent-v1`.

Generate and verify this matrix explicitly:

```bash
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_latent_proxy_recipes.yaml
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_latent_proxy_recipes.yaml --check
unset LV_INITIALIZE_FROM
bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/tiny_latent_proxies_v1.queue.tsv 8
```

The first queue row builds or validates the taxonomy atomically. The remaining
rows independently smoke, sweep `16,32,64,96,128,160`, and train these eight
scratch models to 10k: `latent_slow_proxy`, `latent_fast_proxy`,
`latent_soft_onset_proxy`, `latent_hard_onset_proxy`, `latent_static_proxy`,
`latent_moving_proxy`, `latent_smooth_proxy`, and `latent_irregular_proxy`.
Every config intersects its bucket allowlist with Arp, Bass, FX, Lead, Pad,
Pluck, and Synth, uses a distinct run root, and writes checkpoints every 1k;
therefore 1k, 5k, and 10k weights are retained for each proxy family.

After every model finishes its smoke, sweep, and full 10k train row, the queue
runs 24 `audition_report` rows: 1k, 5k, and 10k for each of the eight proxy
families. The renderer filters held-out rows by the same preset allowlist used
for training, checks that every selected `canonical_preset_id` belongs to that
bucket, and records the allowlist SHA-256 in the audition manifest. Keeping all
training rows before all reports means a render or evaluation failure cannot
hide any later planned weight.

## Tiny audio-descriptor proxy experiment matrix

The third pure specialist matrix uses proxy scores computed from authoritative
raw Serum renders rather than from latent trajectories. Its declarative
artifacts are:

- recipe: `scripts/lvzihao/tiny_audio_proxy_recipes.yaml`;
- generated configs:
  `configs/zrave/generated/lvzihao_tiny_audio_proxies_v1/`;
- training/evaluation queue:
  `scripts/lvzihao/tiny_audio_proxies_v1.queue.tsv`;
- independent report-only CLAP queue:
  `scripts/lvzihao/tiny_audio_proxies_v1.clap.queue.tsv`;
- required imported taxonomy:
  `$HOST_FLOW_ROOT/taxonomies/serum128-audio-v1`.

Do not submit this queue until the taxonomy has been built and sealed on
Octopus using the exact commands in `docs/ZRAVE_TIMBRE_TAXONOMY.md`, then
synchronized as a complete immutable directory. The 5080 source subset is not
the full authoritative corpus, and `taxonomy_validate_audio` deliberately
fails rather than calculating or replacing a missing import.

Generate and verify the deterministic matrix with:

```bash
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_audio_proxy_recipes.yaml
python scripts/lvzihao/generate_category_matrix.py \
  --recipes scripts/lvzihao/tiny_audio_proxy_recipes.yaml --check
unset LV_INITIALIZE_FROM
bash scripts/lvzihao/submit_queue.sh \
  scripts/lvzihao/tiny_audio_proxies_v1.queue.tsv 8
```

The first row validates the Octopus export against the current pack and its
manifest/content provenance. Only after that succeeds, all eight independent
scratch families run smoke, the `16,32,64,96,128,160` batch sweep, and a 10k
train with segment-2/4/8 sampling and checkpoints every 1k:
`audio_dark_proxy`, `audio_bright_proxy`, `audio_soft_attack_proxy`,
`audio_hard_attack_proxy`, `audio_static_proxy`, `audio_moving_proxy`,
`audio_clean_proxy`, and `audio_noisy_proxy`. Every config intersects its
allowlist with Arp/Bass/FX/Lead/Pad/Pluck/Synth and has a separate run root.
All 24 allowlist-filtered 1k/5k/10k `audition_report` rows remain after every
weight-production row; the optional 24 CLAP reports are in the separate queue.

These family names are relative audio-descriptor proxies, not verified product
labels. Dark/bright uses normalized spectral centroid, clean/noisy uses
spectral flatness, and static/moving uses framewise energy/centroid change; the
latter is not literal musical event density. Soft/hard attack is an early RMS
rise proxy, not a complete ADSR envelope and especially not Release, because
the source contract has no authoritative note-off/tail annotation.

## Optional frozen-CLAP audio/audio monitor

`midibrave.zrave_clap_monitor` reuses the repository's
`FrozenClapReconstructionObjective` preprocessing/embedding path and the
existing finite `(512,)` CLAP cache validation contract. For each audition
triplet it reports three audio/audio cosines:

- generated versus source;
- generated versus RAVE Direct;
- RAVE Direct versus source, treated as the codec ceiling.

Each summary contains count, P10, median, P90, worst cosine, and worst row ID.
The codec ceiling deduplicates repeated source/direct pairs across generation
seeds. The report binds the audition manifest, every WAV, every cached
embedding, the CLAP config, and the checkpoint by SHA-256. Audio embeddings are
cached under `clap-monitor-cache/v1/clap/<audio-sha256>.npy`; the JSON sidecar
also binds the preprocessing seed, config/checkpoint hashes, and embedding
hash. Long-audio `rand_trunc` is made reproducible by seeding each one-item
embedding call from its source-audio hash. Each call saves, seeds, and restores
Python, NumPy, Torch CPU, and Torch CUDA RNG states, so evaluation order cannot
change an embedding and the monitor cannot perturb a caller's RNG stream.

This metric is report-only. No caption or text embedding is evaluated, so the
report explicitly does **not** claim text-semantic accuracy or perceptual
timbre identity. It has no hard threshold until source/RAVE Direct baselines
have established a calibrated distribution.

The only recommended checkpoint is the existing authoritative Octopus asset:

```text
/data/model_weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt
bytes: 2352471003
sha256: fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd
```

Synchronize it to
`$HOST_FLOW_ROOT/model-weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt`.
The evaluator itself enforces filename, byte size, and SHA-256; an uncalibrated
`timbreclap-v2` checkpoint is not accepted.

The normal `midibrave:lvzihao-cu128-v1` image intentionally stays CLAP-free.
Build the opt-in derivative with `scripts/lvzihao/build_clap_image.sh`. The
builder first inspects the selected base image and binds its exact Torch,
torchaudio, and torchvision versions into the build assertions; this supports
both the pinned 2.9.1 base and the audited 2.10 fallback without silently
mixing wheels. Its
userspace matches the authoritative Octopus CLAP environment:
`laion-clap==1.1.7`, `transformers==5.13.0`, `torchlibrosa==0.1.0`,
`librosa==0.11.0`, and `ftfy==6.3.1`, while retaining lvzihao's audited
Torch/CUDA 12.8 base and its matching torchaudio/torchvision builds. The
package requires additional transitive dependencies
including braceexpand, webdataset, wget, progressbar, resampy, pandas, h5py,
scikit-learn, and wandb. The CLAP checkpoint itself is about 2.19 GiB; package
and tokenizer layers are additional. For capacity planning, expect roughly
0.5--1.0 GiB of additional expanded Python packages in the derived image and
about 0.5 GiB of external Hugging Face assets, separately from the 2.19 GiB
checkpoint. These are estimates, not an artifact contract: record the actual
Docker image and cache sizes after the first build/sync.

Although this monitor never calls a text embedding, `laion_clap 1.1.7` creates
tokenizers at module-import time. An offline Octopus import was observed to
fail first on `bert-base-uncased`; the same source module then initializes
`roberta-base` and `facebook/bart-base`, while `CLAP_Module` constructs the
RoBERTa text branch. Therefore the mounted `$HF_HOME` must contain the complete
`bert-base-uncased` tokenizer, `roberta-base` tokenizer and model, and
`facebook/bart-base` tokenizer before the offline qgpu run. The action checks
all three with `local_files_only=True` before importing CLAP. Missing
checkpoint, CLAP-capable image, or any HF cache asset is a reason to skip the
optional report, never a reason to stop training queues.

Run it only in its own queue after an audition exists:

```text
# id  action       config  run  spec
clap-pad-010000  clap_report  -  -  tinycat-v1-pad-audition-010000
```

The matrix generator writes these independent queues automatically for all
four current matrices: `tiny_categories_v1.clap.queue.tsv`,
`tiny_latent_proxies_v1.clap.queue.tsv`,
`tiny_audio_proxies_v1.clap.queue.tsv`, and
`tiny_midi_categories_v1.clap.queue.tsv`. They contain only `clap_report` rows
and are never submitted as part of weight training.

Set `LV_IMAGE=midibrave:lvzihao-cu128-clap-v1` and
`LV_CLAP_CHECKPOINT=.../music_audioset_epoch_15_esc_90.14.pt` before submitting.
The contract fixes embedding batch size to 1 for per-audio deterministic crop
seeding. Actual peak CUDA allocated bytes/GiB are written to every report; on
the 16 GiB 5080, do not raise the batch until a real qgpu smoke establishes a
safe margin. No measured lvzihao peak is claimed before that run.

## References

- [PyTorch 2.9 release](https://pytorch.org/blog/pytorch-2-9/)
- [Official PyTorch CUDA compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
- [Pinned official linux/amd64 image manifest](https://hub.docker.com/layers/pytorch/pytorch/2.9.1-cuda12.8-cudnn9-runtime/images/sha256-7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385)
