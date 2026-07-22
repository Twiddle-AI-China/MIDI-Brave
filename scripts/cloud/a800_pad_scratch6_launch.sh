#!/usr/bin/env bash
set -Eeuo pipefail

export MIDIBRAVE_REPO=/data/run01/scwc257/latent-cosmos-synth/MidiBrave-v3-pad-scratch-20260722/MidiBrave-v2
export MIDIBRAVE_CLAP_CHECKPOINT=/data/run01/scwc257/latent-cosmos-synth/model_weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt

exec bash "$MIDIBRAVE_REPO/scripts/cloud/a800_pad_scratch6_curriculum.sh"
