#!/bin/bash
# SLEAP pose estimation into DataJoint (pose.SleapRawTracking), one experiment.
#
# Adapted from submodules/foragingABC_analysis/slurm/run_pose_per_experiment.sh.
# Every array task runs the same populate; DataJoint job reservation hands
# each task different chunks. Inference uses the legacy sleap_legacy conda env
# (see scripts/pipeline/sleap_legacy.py); the main .venv runs DataJoint.
#
# Submit from the repo root:
#   mkdir -p logs && sbatch scripts/hpc/submit_pose.sh ProjectAeonOVC-2026-09-14_001
#
#SBATCH -J ovc_pose
#SBATCH -p gpu_branco
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-08:00
#SBATCH --array=1-5
#SBATCH -o logs/pose_%A_%a.out
#SBATCH -e logs/pose_%A_%a.err

EXPERIMENT="${1:?usage: sbatch scripts/hpc/submit_pose.sh <experiment_name>}"
PROCESSES=1      # chunks per GPU; raise once GPU memory sharing is confirmed
BATCH_SIZE=8
MAX_CALLS=20     # chunks per task before it exits; resubmit for the rest

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_DIR/.venv/bin/python}"
cd "$REPO_DIR" || exit 1

echo "Experiment: $EXPERIMENT | task ${SLURM_ARRAY_TASK_ID:-1} | node $(hostname) | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

"$VENV_PYTHON" scripts/pipeline/populate_pose.py "$EXPERIMENT" \
    --processes "$PROCESSES" \
    --batch-size "$BATCH_SIZE" \
    --max-calls "$MAX_CALLS" \
    --max-wait 60

EXIT_CODE=$?
echo "Finished: $(date) | exit $EXIT_CODE"
exit $EXIT_CODE
