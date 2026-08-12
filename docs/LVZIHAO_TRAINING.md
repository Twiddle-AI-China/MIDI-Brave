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
`LV_INITIALIZE_FROM`, hashes the selected flow checkpoint and 85k initializer,
renders matched and observed-note-swap audio, then applies the generic 128D
long-rollout degeneration gate. A rejected `gate.json` blocks the remaining
audition rows. This is a training-plus-audio-health gate only; quantitative
MIDI pitch adherence for matched and 36 -> 62 -> 82 -> 36 swapped controls
remains to be implemented before G3 can pass its MIDI adherence gate.

## Queue and resume behavior

The queue is a five-column, tab-separated file:

```text
experiment_id  action  config_relative  run_relative  spec
```

Actions and `spec` values are:

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
- `audition`: `latest`, `final.pt`, or an explicit `step-N.pt` basename. Output
  is promoted from a partial directory only after every referenced WAV is
  finite and has the expected shape/rate.
- `midi_audition`: the same checkpoint spec. It strictly validates the MIDI
  architecture, checkpoint/initializer/pitch-probe/config/data hashes, renders
  the six test categories, and promotes output only when the generic
  degeneration gate passes. It does not yet claim MIDI adherence.

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

## References

- [PyTorch 2.9 release](https://pytorch.org/blog/pytorch-2-9/)
- [Official PyTorch CUDA compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
- [Pinned official linux/amd64 image manifest](https://hub.docker.com/layers/pytorch/pytorch/2.9.1-cuda12.8-cudnn9-runtime/images/sha256-7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385)
