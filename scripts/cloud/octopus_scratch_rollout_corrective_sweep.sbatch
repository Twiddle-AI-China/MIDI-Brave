#!/usr/bin/env bash
#SBATCH --job-name=mb-scratch-corr-sweep
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/data/midibrave-v3/launches/scratch-pad-corrective-20260722/logs/sweep-%j.out

set -euo pipefail

repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
launch=/data/midibrave-v3/launches/scratch-pad-corrective-20260722
image=midibrave:v0.10.0-clap-recon
template="$repo/configs/v3/octopus_scratch_pad_rollout_corrective.yaml"
checkpoint=/data/midibrave-v3/scratch-pad-final-audition/input/update-00001500.pt
statistics=/data/midibrave-v3/cache/scratch-pad-20260722/rave-statistics.npz
calibration=/data/midibrave-v3/cache/scratch-pad-20260722/predictive-calibration.json
sweep_root="/data/midibrave-v3/sweeps/scratch-pad-corrective-${SLURM_JOB_ID}"
summary="$sweep_root/summary.tsv"
candidates=(${CANDIDATES:-32 40 44 48})

mkdir -p "$launch/logs" "$sweep_root"
printf 'batch_per_gpu\tstatus\telapsed_seconds\tsamples_per_second\tpeak_memory_mib\ttotal_memory_mib\tevent\n' > "$summary"

for batch in "${candidates[@]}"; do
  run_name="scratch_pad_corrective_sweep_b${batch}_${SLURM_JOB_ID}"
  config="$sweep_root/config-b${batch}.yaml"
  monitor="$sweep_root/gpu-b${batch}.csv"
  run_dir="/data/midibrave-v3/runs/${run_name}/rollout"
  sed \
    -e "s/run_name: pad50_scratch_rollout_corrective_v100_8gpu/run_name: ${run_name}/" \
    -e 's/rollout_steps: 100000/rollout_steps: 2/' \
    -e 's/checkpoint_every: 2000/checkpoint_every: 2/' \
    -e 's/log_every: 20/log_every: 1/' \
    "$template" > "$config"

  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "$monitor" 2> "$monitor.err" &
  monitor_pid=$!
  started=$(date +%s)
  set +e
  srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" --shm-size=32g --ulimit memlock=-1 \
    -v "$repo":/workspace/MidiBrave:ro -v /data:/data \
    -e PYTHONPATH=/workspace/MidiBrave/src \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -w /workspace/MidiBrave "$image" \
    torchrun --standalone --nproc_per_node=8 -m midibrave.trainer \
    --config "$config" --stage rollout --max-steps 2 \
    --batch-per-gpu "$batch" --latent-statistics "$statistics" \
    --calibration "$calibration" --warm-start "$checkpoint"
  status=$?
  set -e
  elapsed=$(( $(date +%s) - started ))
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  event=$(find "$run_dir" -maxdepth 1 -type f -name 'events.out.tfevents.*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
  samples_per_second=NA
  if [[ -n "$event" ]]; then
    samples_per_second=$(docker run --rm -v /data:/data "$image" python -c \
      'import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; event=EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); event.Reload(); values=event.Scalars("train/samples_per_second"); print(values[-1].value if values else "NA")' \
      "$event")
  fi
  peak_memory=$(awk -F, 'BEGIN { peak=0 } { value=$3+0; if(value>peak) peak=value } END { print peak }' "$monitor")
  total_memory=$(awk -F, 'NF>=4 { print $4+0; exit }' "$monitor")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$batch" "$status" "$elapsed" "$samples_per_second" \
    "${peak_memory:-NA}" "${total_memory:-NA}" "${event:-NA}" | tee -a "$summary"
done

cat "$summary"
