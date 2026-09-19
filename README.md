# MidiBrave

Independent dual-branch MIDI-conditioned BRAVE training framework.

## Current Atlas Flow v5 handoff

The current Pad Top50 trajectory model, evaluation portal, and continuous live
instrument are implemented in `src/midibrave/atlas_flow_*.py`,
`configs/atlas_flow/`, and `atlas-flow-live-dashboard/`. GPU inference is a
Dockerized SLURM workload on Spark; Octopus is retained only as the training
artifact source. The complete architecture, data, training, evaluation,
operations, limitations, and recovery notes are maintained in
`docs/Atlas-Flow-v5-完整交接文档-2026-08-26.md`.

## Atlas Flow desktop app (v1)

A self-contained desktop build of the Pad Top50 instrument: the timbre map, live
roaming, chord progressions, drawn trajectories and the three scopes, with the
model running on the machine in front of you. No cluster, no GPU, no tunnel, no
Python to install.

### Getting it

Download the build for your platform from the **Atlas Flow desktop** workflow's
artifacts, unzip, and run it.

| | |
|---|---|
| macOS (Apple silicon) | `AtlasFlow.app` — right-click → **Open** the first time. The build is unsigned, so a double-click gets refused with "unidentified developer"; opening it from the context menu once is enough, and macOS remembers. |
| Windows (x64) | `AtlasFlow.exe` inside the unzipped folder. SmartScreen will warn for the same reason: **More info → Run anyway**. Keep the folder together — the .exe needs the files beside it. |

It is about 1 GB unzipped, most of which is torch. The **first launch takes
30–60 seconds** while the model and its runtime are read off disk for the first
time; the window says so while it waits. Later launches take a few seconds.

Where it puts things:

| | macOS | Windows |
|---|---|---|
| Renders, takes, recordings | `~/Library/Application Support/AtlasFlow/model-cache` | `%APPDATA%\AtlasFlow\model-cache` |
| Launch log | `~/Library/Application Support/AtlasFlow/desktop.log` | `%APPDATA%\AtlasFlow\desktop.log` |

If the window says the server did not come up, that log is what to send.

It picks the fastest backend it finds: CUDA, then Metal, then CPU. CPU works —
measured on an M2 a five-second plan takes ~570 ms against ~290 ms on Metal, and
sustaining a note costs nothing either way, because the runtime time-warps a
plan it already has rather than generating continuously.

### Running from a checkout

For the same thing without packaging, needing only Python:

```bash
python -m pip install torch soundfile numpy scipy aiohttp PyYAML
bash scripts/atlas_flow/fetch_local_model.sh    # once: 85 MB weights + atlas + evaluation
python -m midibrave.atlas_flow_local            # then: http://127.0.0.1:18796
```

`atlas_flow_local` is the cross-platform entry point and takes `--port`,
`--device`, `--model` and `--threads`; run it with `--help`. On Windows use it
directly — `local_demo.sh` is a thin bash convenience wrapper around it, nothing
more. Set `PYTHONPATH=src` if the package is not installed.

To run the desktop shell against a checkout rather than a packaged build:

```bash
npm --prefix electron install
npm --prefix electron start
```

### Building the app

Both halves are built on the machine that will run them: the frozen server
contains native torch libraries, so a macOS host produces the macOS app and a
Windows host the Windows one. There is no cross-compiling.

```bash
python -m pip install torch soundfile numpy scipy aiohttp PyYAML pyinstaller
npm --prefix electron install
npm --prefix electron run build          # freeze the server, then package the app
```

`build` is the two steps together; run `build:server` and `pack` separately if
you are iterating on one of them. The result lands in `electron/dist/`.

By default the app is built around `local-model/`. Point it at a different set
of weights with `ATLAS_PACK_MODEL`:

```bash
ATLAS_PACK_MODEL=/path/to/weights npm --prefix electron run build
```

### Testing it

Two test suites, both of which CI runs on macOS and Windows:

```bash
python packaging/smoke_test.py --server build/pyi/atlas-flow-server/atlas-flow-server
node electron/smoke.js
```

The first exercises the frozen server: the atlas loads with all 50 presets, the
web assets are served from inside the bundle, a render produces audio that is
not silence, the same seed reproduces the same bytes, a three-note chord sums
three voices, and the live websocket negotiates a format and delivers PCM.

The second asks the narrower question only a packaged build can answer — is the
app actually self-contained? It runs the server from inside the app with no
`PYTHONPATH`, no virtualenv and the working directory set to the filesystem
root, so anything still reaching for the checkout fails there. It is pure Node
on purpose: a test that needed Python could not tell the difference.

To test without the trained weights — which is what CI does, since the real ones
live on a cluster no runner can reach:

```bash
python packaging/make_test_model.py --out build/test-model
ATLAS_PACK_MODEL=$PWD/build/test-model npm --prefix electron run build
```

That builds the same architecture from the same config with random parameters.
Everything downstream behaves identically and the audio is noise, which is the
point: it proves the machinery works on a platform without claiming anything
about how the model sounds.

## Current v2 qualification rollout

The active implementation is under `configs/v2/` and `scripts/v2/`: 44.1 kHz,
49,152-sample variable-valid windows, 256-D timbre latent, 32-D MIDI condition,
decoder capacity 64, stochastic 16-band excitation, bounded FiLM and
FP32-reduction RMSNorm. The generator has 8,175,936 parameters and the
official-size BRAVE discriminator has 1,940,451.

Selection is deterministic and mutually exclusive: 400 timbres each for Pad,
Lead, Base, Pluck and tonal Texture, with Serum > Dexed v2 > velocity-known
sample v2 priority and QA score `0.8*label + 0.2*CLAP`. The first qualification
trains Lead only, comparing `safe_fallback` with `fp16_candidate` at 200, 1,000
and 5,000 effective updates on 2xV100. It does not launch the other classes,
Phase 2, or a 20k run.

The model is conditioned only on MIDI note and velocity. Full Serum renders are
retained for CLAP, QA, and fixed condition-level velocity RMS references. Decoder targets use 49,152-sample (about 1.115 s)
windows, zero-padding genuinely short recordings while tracking exact valid lengths; unreliable frames are masked only from pitch loss,
not removed from reconstruction. The host owns the performance envelope.
Training reconstructs two phase-agnostic outputs:

```text
A_hat = Decoder(z_timbre_A, z_midi_A)
B_hat = Decoder(z_timbre_A, z_midi_B)
```

The current CLAP-reconstruction profile keeps the frozen complete-render CLAP
conditioning path and the 256-D TimbreAdapter output. It additionally compares
frozen CLAP embeddings of the exact generated and target reconstruction windows
for both Self and Cross. Target embeddings are evaluated without gradients;
generated windows preserve a first-order waveform gradient through CLAP. To
avoid retaining HTSAT-base and full-batch decoder activations together, the
trainer computes the sampled CLAP waveform gradient in a no-grad decoder
prepass, releases CLAP activations, and injects that gradient into the matching
regular decoder output. The default profile samples one pair per GPU every four
updates with inverse-probability correction and a 1k weight warm-up.

MIDI note also drives a parameter-free harmonic excitation clock. Its 16-band
PQMF pyramid is injected at every BRAVE upsampling scale, while velocity remains
a learned part of the 32-D MIDI condition. Losses use fullband/PQMF MR-STFT,
envelope dynamics, frozen differentiable CREPE with absolute target activation,
hard-negative and target-period autocorrelation constraints, generated-waveform
dB RMS, masked same-note velocity ranking, and optional observed velocity
dB-delta matching (disabled in the validated C9 profile).
Window reconstruction losses always use the sampled windows; only the relative
velocity target uses complete-render RMS so independently sampled crop offsets
cannot flip its label. There is no sample waveform loss or voiced loss.

`condition_gain_hidden > 0` enables an experimental, zero-initialized bounded
output-gain head. It is disabled by default: the C14/C15 ablation did not make
arbitrary non-monotonic velocity responses generalize to unseen presets and
increased crest/ripple error. No long training should be launched until the
product chooses between exact preset-specific velocity response and a unified
real-time MIDI velocity semantic.

Phase 2 uses the official-size three-scale BRAVE waveform discriminator
(1,940,451 parameters). The trainer counts only finite, jointly applied G/D
updates and stores exact-resume checkpoint format 4, including scheduler,
precision, sampler and RNG state.

The historical v1 profiles lock the validated C9 pitch-repair loss:
`cross_stft=0.5`, `cross_pitch=1.0`, KL off, target activation 1.0,
hard-negative 0.25, target-period autocorrelation 20.0, velocity rank 0.5,
and velocity delta off. Use `configs/full_c9_optimized.yaml` for the full
16+2-epoch run and `configs/quality300_c9_optimized.yaml` for the matching
q300 view. The quality-first full-compute profile is stored separately under
`training_profiles/pre_time_optimization_c9/`: it keeps the same C9 loss and
algebra-preserving speedups, but restores Self on every update and the fixed
1,000,000 + 250,000 update budget.

The v2 profiles retain the stable C9 basis but reduce velocity ranking to 0.05,
add an every-step analytic pitch objective and isolate sparse frozen-CREPE
gradients behind a sanitized, per-sample clipped boundary. See
`configs/v2/generated/lead_safe_fallback.yaml` for the current contract.

## Quick smoke

```bash
midibrave fixture --root fixtures/generated
midibrave validate --config configs/smoke.yaml
midibrave-train --config configs/smoke.yaml --phase 1 --max-effective-updates 4
midibrave-train --config configs/smoke.yaml --phase 2 --max-effective-updates 3 \
  --resume artifacts/candidate/smoke_continuous_v02/phase1/step-000000004.pt
```

GPU work on Octopus must run through SLURM and Docker. See `scripts/`.

## Pure Z-RAVE Transformer + Flow Matching POC

The active proof of concept learns stochastic continuations of the original
16-D RAVE latent sequence. It consumes 32 consecutive `z_rave` frames and uses
a Transformer velocity field with Flow Matching to generate the next 64
frames. This path has no MIDI, pitch probe, CLAP, waveform reconstruction, or
KL objective. Its loss is flow matching plus small boundary-continuity and
latent-statistics terms.

The completed pack at
`/data/midibrave-zrave-flow/packs/all-synth` already contains Serum,
Pianobook, and Dexed/Surge material. Do not encode the source audio again.
Submit only the throughput sweep and formal training:

```bash
sweep_job=$(sbatch --parsable \
  scripts/cloud/octopus_zrave_pure_flow_sweep.sbatch)
train_job=$(sbatch --parsable --dependency=afterok:"$sweep_job" \
  scripts/cloud/octopus_zrave_pure_flow_train.sbatch)
```

The sweep measures per-GPU batches 128, 256, 384, and 512 for 100 updates on
eight GPUs and selects the highest valid latent-frame throughput. Formal
training runs 100,000 updates and writes a checkpoint every 5,000 updates under
`/data/midibrave-zrave-flow/runs/pure-flow-poc-v1`; TensorBoard events are in
the adjacent `tensorboard/` directory.

Runtime needs only a saved 32-frame latent seed, the pure-flow checkpoint,
generation seed, temperature, wander delay, and the frozen RAVE decoder. The
RAVE encoder is not required after seed-bank construction. The older
MIDI-conditioned flow scripts remain solely for artifact compatibility and are
not part of this POC.

### Serum Balanced 128-D predictor

The current paired experiment replaces the historical 16-D Freesound codec
with the Serum Balanced RAVE trained in this repository. It is bound to the
recoverable Phase 2 checkpoint at global step `968805` (SHA-256
`3a158421...bde7`). The derived runtime codec keeps all 128 PCA-rotated
posterior-mean coordinates instead of applying RAVE's post-hoc fidelity crop;
its SHA-256 is `69e8a2a1...f9b4`.

This remains a pure `z_rave` proof of concept: 32 real frames condition a
Transformer velocity field that samples the next 64 frames with conditional
Flow Matching. There is no MIDI, CLAP, pitch, waveform, or KL objective. The
source checkpoint, config, corpus manifest, and Serum audio are mounted
read-only. Derived latents and predictor artifacts live only under
`/data/midibrave-zrave-flow-serum128`.

Submit the complete Octopus chain with dependencies:

```bash
prepare=$(sbatch --parsable \
  scripts/cloud/octopus_serum128_flow_prepare.sbatch)
sweep=$(sbatch --parsable --dependency=afterok:"$prepare" \
  scripts/cloud/octopus_serum128_flow_sweep.sbatch)
train=$(sbatch --parsable --dependency=afterok:"$sweep" \
  scripts/cloud/octopus_serum128_flow_train.sbatch)
```

Preparation encodes the exact Serum Balanced split into a new 128-D pack. The
sweep measures per-GPU batches 64, 96, 128, and 160 for 100 updates each on
eight V100s. Formal training selects the highest valid latent-frame throughput,
runs 100,000 updates, and saves every 5,000 updates under
`runs/pure-flow-v1/checkpoints`; TensorBoard events are under the adjacent
`tensorboard/` directory.

### Exploration-aware Serum128 continuation

`octopus_serum128_exploration_flow.yaml` is the next pure Z-RAVE experiment;
it still contains no MIDI, CLAP, or audio reconstruction path. Training starts
from the 85k pure-flow model weights but deliberately resets optimizer, AMP,
RNG, and update state. Its objective is Flow Matching plus boundary (0.10),
latent-statistics (0.02), and per-sample multi-scale temporal motion (0.05)
terms. A ramped depth-0-to-3 curriculum replaces up to 48 history frames with
the model's own 16-frame commits, so recursive errors are visible during
training instead of only during audition.

At runtime one exploration value maps to visible history (32 down to 8
frames), sampling temperature (0.7 up to 1.3), and wander delay (48 down to 16
frames). Each step samples four 64-frame candidates, rejects stagnant,
boundary-jumping, or out-of-range prefixes, commits only the best first 16
frames, and advances one absolute schedule. The model shape remains
`[B,32,128] -> [B,64,128]`; the shorter commit is a rollout policy.

Run the required 8-GPU sweep and training through SLURM:

```bash
sweep=$(sbatch --parsable \
  scripts/cloud/octopus_serum128_exploration_sweep.sbatch)
train=$(sbatch --parsable --dependency=afterok:"$sweep" \
  scripts/cloud/octopus_serum128_exploration_train.sbatch)

# Optional explicit overrides; automatic resume wins when a v2 checkpoint exists.
sbatch --export=ALL,BATCH_PER_GPU=160,MAX_UPDATES=20000 \
  scripts/cloud/octopus_serum128_exploration_train.sbatch
```

The sweep measures per-GPU batches 128, 160, 192, and 224 with 10 warm-up,
100 timed, and five forced depth-three updates. Checkpoints are written every
5,000 updates under `runs/exploration-v2/checkpoints`. Inspect
`train/{flow,boundary,statistics,temporal}`, `health/exposure_depth`,
`train/visible_history_frames`, and `train/schedule_offset_frames` in
TensorBoard:

```bash
tensorboard \
  --logdir /data/midibrave-zrave-flow-serum128/runs/exploration-v2/tensorboard \
  --host 127.0.0.1 --port 6006

sbatch --export=ALL,SERUM128_EXPLORATION_AUDITION_CHECKPOINT=step-020000.pt \
  scripts/cloud/octopus_serum128_exploration_audition.sbatch
```

The audition job renders held-out Pad, Lead, Bass, Pluck, Keys, and Synth at
exploration 0.0, 0.5, and 1.0. Its manifest records the checkpoint and 85k
initializer hashes, selected candidate indices, motion, boundary, and latent
range diagnostics for every 16-frame commit.

## Predictive RAVE + CLAP runtime (v3)

The v3 path restores a causal RAVE encoder during training, concatenates its
16-D latent with user-controlled 256-D CLAP and 32-D MIDI controls, and learns
an eight-frame continuation head. Each runtime call consumes four frames
(`4 * 128 = 512` samples) while retaining a 16-frame history. Production export
contains no RAVE encoder and no CLAP audio encoder: it starts from a validated
training-data seed bank and continues the RAVE latent sequence directly.

Run the stages in order. Replace checkpoint names and `<RAVE_SHA256>` with the
actual archived artifact and its SHA-256 digest:

```bash
midibrave fixture --root fixtures/generated --samples 65536
midibrave preprocess --config configs/v3/smoke.yaml --stage all --device cuda \
  --clap-checkpoint /data/model_weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt

midibrave-train --config configs/v3/smoke.yaml --stage rave
midibrave cache-rave --config configs/v3/smoke.yaml \
  --checkpoint artifacts/predictive-smoke/predictive_rave_v1_smoke/rave/update-00000004.pt
midibrave build-seed-bank --config configs/v3/smoke.yaml \
  --checkpoint-hash <RAVE_SHA256> --output artifacts/predictive-smoke/seed-bank

midibrave-train --config configs/v3/smoke.yaml --stage predictor \
  --warm-start artifacts/predictive-smoke/predictive_rave_v1_smoke/rave/update-00000004.pt
midibrave-train --config configs/v3/smoke.yaml --stage rollout \
  --warm-start artifacts/predictive-smoke/predictive_rave_v1_smoke/predictor/update-00000004.pt
midibrave-train --config configs/v3/smoke.yaml --stage gan \
  --warm-start artifacts/predictive-smoke/predictive_rave_v1_smoke/rollout/update-00000004.pt \
  --rollout-gate-passed

midibrave-evaluate --config configs/v3/smoke.yaml \
  --checkpoint artifacts/predictive-smoke/predictive_rave_v1_smoke/rollout/update-00000004.pt \
  --output artifacts/predictive-smoke/evaluation --pairs 2
midibrave export-predictive --config configs/v3/smoke.yaml \
  --checkpoint artifacts/predictive-smoke/predictive_rave_v1_smoke/gan/update-00000003.pt \
  --seed-bank artifacts/predictive-smoke/seed-bank \
  --output artifacts/predictive-smoke/runtime.pt
```

The predictor stage automatically calibrates fixed loss weights from the
configured calibration batches. Latent, delta, acceleration, and overlap losses
must use the future of the same recording; cross-recording latent MSE is invalid
because phase and texture trajectories are not aligned. Rollout evaluation
reports 1/8/32/128-step quality metrics, but gates only numerical safety,
variance collapse/explosion, the 512-sample stride, and underruns until measured
quality baselines are established.

Build the image and cache frozen preprocessing models under `/data/model_weights`:

```bash
./scripts/build_image.sh
./scripts/download_models.sh
```

## Published Serum data

The source dataset is immutable. Build strict derived views, cache frozen
features, and freeze a cache-eligible manifest before training:

```bash
midibrave prepare-serum \
  --dataset-root /data/datasets/latent-cosmos-synth/serum-dataset \
  --output /data/midibrave/manifests
midibrave preprocess --config configs/quality300.yaml --stage all --device cuda
midibrave finalize-cache --config configs/quality300.yaml \
  --output-manifest /data/midibrave/manifests/serum_quality300_eligible_optimized.jsonl \
  --output-metadata /data/midibrave/manifests/serum_quality300_eligible_optimized.meta.json
```

The older `serum_quality300_eligible.jsonl` is intentionally retained as the
65,536-sample/15%-valid historical comparison and must not be paired with a
49,152-sample/75%-valid optimized config.

Audio conditioning always uses the audited final `midi_note`; `midi_note_sent`
and `transpose_semitones` are retained only for provenance checks. Missing grid
cells are not negatives and no 72/72 grid is assumed.
