#!/usr/bin/env bash
#SBATCH --job-name=mb-p3-sweep
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/data/midibrave-v3/launches/pad50-20260721/logs/phase3-sweep-%j.out

set -euo pipefail
repo=/home/yfhuang/Latent-Cosmos-Synth-midiBrave/MidiBrave-v2
image=midibrave:v0.10.0-clap-recon
template="$repo/configs/v3/octopus_pad50_phase3_sweep.yaml"
checkpoint=/data/midibrave-v3/runs/pad50_phase2c_control_recovery_from_p2b_3k/predictor/update-00001000.pt
statistics=/data/midibrave-v3/cache/pad50-phase15-5k/rave-statistics.npz
calibration=/data/midibrave-v3/cache/pad50-phase15-5k/predictive-calibration-phase2b.json
sweep_root="/data/midibrave-v3/sweeps/phase3-${SLURM_JOB_ID}"
summary="$sweep_root/summary.tsv"
candidates=(${CANDIDATES:-16 24 32 40})

mkdir -p "$sweep_root"
printf 'batch_per_gpu\tstatus\telapsed_seconds\tsamples_per_second\tnonfinite_updates\tpeak_memory_mib\ttotal_memory_mib\tevent\n' > "$summary"

for batch in "${candidates[@]}"; do
  run_name="pad50_phase3_batch_sweep_b${batch}_${SLURM_JOB_ID}"
  config="$sweep_root/config-b${batch}.yaml"
  monitor="$sweep_root/gpu-b${batch}.csv"
  run_dir="/data/midibrave-v3/runs/${run_name}/rollout"
  sed "s/run_name: pad50_phase3_batch_sweep/run_name: ${run_name}/" \
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
    -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -w /workspace/MidiBrave "$image" \
    torchrun --standalone --nproc_per_node=8 -m midibrave.trainer \
    --config "$config" --stage rollout --max-steps 6 \
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
  nonfinite_updates=NA
  if [[ -n "$event" ]]; then
    read -r samples_per_second nonfinite_updates < <(
      docker run --rm -v /data:/data "$image" python -c \
      'import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; event=EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); event.Reload(); scalar=lambda name: event.Scalars(name)[-1].value if event.Scalars(name) else float("nan"); print(scalar("train/samples_per_second"), scalar("health/nonfinite_updates"))' \
      "$event")
  fi
  peak_memory=$(awk -F, 'BEGIN { peak=0 } { value=$3+0; if(value>peak) peak=value } END { print peak }' "$monitor")
  total_memory=$(awk -F, 'NF>=4 { print $4+0; exit }' "$monitor")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$batch" "$status" "$elapsed" "$samples_per_second" "$nonfinite_updates" \
    "${peak_memory:-NA}" "${total_memory:-NA}" "${event:-NA}" | tee -a "$summary"
done

cat "$summary"
