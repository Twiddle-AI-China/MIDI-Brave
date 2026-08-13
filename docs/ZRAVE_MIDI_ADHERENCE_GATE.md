# G3 MIDI latent-probe and decoded-audio gates

There are two deliberately separate MIDI adherence measurements. They must
never be relabeled as one another.

## Latent pitch-probe proxy

`midibrave-zrave-midi-adherence-gate` checks the G3 audition manifest after
rendering. It is deliberately named a `latent_pitch_probe_proxy`: a qualified,
frozen pitch probe evaluates non-overlapping 16-frame generated latent windows
against the requested MIDI note at each window's lower midpoint.

The report includes matched, observed-vocabulary `note_swap`, and midpoint
`note_step` coverage and
window counts, exact-class accuracy, expected-MIDI absolute cents median/p90,
and within-50/within-100-cent ratios. Hard gates require both control kinds,
finite window results, p90 no greater than 100 cents, at least 90% of windows
within 100 cents, valid probe hashes, and a note-swap mapping that actually
changes every paired requested note.

This is not decoded-audio MIDI accuracy. The latent probe cannot observe
voicing, so voiced coverage is explicitly `null`.

## Decoded-audio CREPE gate

`midibrave-zrave-decoded-audio-midi-gate` is the independent
`decoded_audio_crepe` measurement. It reads the float `raw_wav` for every MIDI
rollout, removes the declared `history_frames * latent_hop` prefix, and tracks
only the generated interval with the official torchcrepe 0.0.24
`torchcrepe.predict` path. Every CREPE frame center is mapped to
`requested_notes[floor(sample / latent_hop)]`.

Matched, `note_swap`, and report-only `note_step` are reported independently.
The hard pitch checks remain bound to matched and note-swap. Each has pooled frame
counts plus per-rollout distributions with count, p10, median, p90, and a
direction-aware worst value for:

- finite analysis frames and voiced frames;
- voiced coverage, whose denominator is every finite in-range CREPE frame;
- voiced-frame absolute cents median/p90 and within-50/100 ratios;
- exact rounded MIDI-note accuracy and octave-error fraction.

An octave error means the nearest nonzero 1200-cent shift has at most a
100-cent residual. Hard checks require both control kinds, a non-empty finite
and voiced measurement, pooled p90 absolute error no greater than 100 cents,
and at least 90% of voiced frames within 100 cents. Voiced coverage is kept
independent and report-only until a threshold is calibrated against held-out
source and direct-RAVE ceilings.

For every change in `requested_notes`, settling is the start of the first run
of three consecutive voiced CREPE frames within 100 cents of the new note. The
history-to-generated boundary is also an event when the first future request
differs from the final `history_requested_notes` value. Settling milliseconds
are report-only. A constant matched control has no event and correctly reports
`not_applicable_no_requested_note_change`; `note_swap` measures boundary
settling and `note_step` measures the recorded-to-next-observed-note transition
at the midpoint of the generated future.

The separate `velocity_step` control keeps note fixed and changes between the
two observed Serum velocities (54 and 108) at the midpoint. Because neither the
latent pitch probe nor CREPE predicts velocity, the report measures steady
pre/post RMS dB change and whether its direction agrees with the requested
velocity direction. This is explicitly a loudness-response proxy, not MIDI
velocity accuracy, and it is report-only.

The report records SHA-256 hashes for the audition manifest, every raw WAV,
the generator checkpoint, RAVE codec, pack index, statistics, audition config,
evaluation config, evaluator module, torchcrepe package code, and embedded
CREPE model. The default official 0.0.24 `tiny.pth` authority is
`d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432`.
Missing packages/assets or a hash mismatch makes the action explicitly
`decoded_audio_crepe blocked`; it never downloads a replacement at runtime.

Build the separate offline image once on lvzihao while package access is
available:

```bash
bash scripts/lvzihao/build_audio_eval_image.sh
```

The normal training image intentionally remains free of torchcrepe. The eval
image is then used under the same qgpu allocation with Docker networking set to
`none`.

On lvzihao, `scripts/lvzihao/midi_audition.sh` always materializes the generic
decoded long-rollout report, latent proxy report, and decoded-audio CREPE
report. Production mode promotes the audition directory only after all three
pass, and retains `gate.json`, `midi-gate.json`, plus
`decoded-audio-midi-gate.json`. Research mode retains a valid negative report
without treating threshold rejection as a queue failure; rendering, manifest
integrity, lineage, and CREPE asset checks remain hard failures.
