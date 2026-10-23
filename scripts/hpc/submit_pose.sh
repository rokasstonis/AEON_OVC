#!/bin/bash
# SLEAP pose estimation into DataJoint (pose.SleapRawTracking), one experiment.
#
# Adapted from submodules/foragingABC_analysis/slurm/run_pose_per_experiment.sh.
# Every array task runs the same populate; DataJoint job reservation hands
# each task different chunks. Inference uses the legacy sleap_legacy conda env
# (see scripts/pipeline/sleap_legacy.py); the main .venv runs DataJoint.
#
# Submit from the repo root (resubmitting is always safe: interrupted chunks are
# re-queued, finished ones skipped, ones running in other tasks left alone):
#   mkdir -p logs && sbatch scripts/hpc/submit_pose.sh ProjectAeonOVC-2026-09-14_001
# Progress:
#   uv run python scripts/pipeline/populate_pose.py ProjectAeonOVC-2026-09-14_001 --status
#
# Parallelism: --array=1-N%M runs up to M tasks (= GPUs) at once. Each task
# keeps taking chunks until none are left or its time budget runs out.
#
#SBATCH -J ovc_pose
#SBATCH -p gpu_branco
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4        # 4 x PROCESSES
#SBATCH --mem=16G                # ~8G x PROCESSES
#SBATCH -t 0-08:00
#SBATCH --array=1-12%6
#SBATCH -o logs/pose_%A_%a.out
#SBATCH -e logs/pose_%A_%a.err

EXPERIMENT="${1:?usage: sbatch scripts/hpc/submit_pose.sh <experiment_name>}"
PROCESSES=1          # chunks per GPU; try 2 once GPU memory use is known
BATCH_SIZE=8
TIME_BUDGET_H=7.5    # keep ~30 min below -t so tasks exit before the limit

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_DIR/.venv/bin/python}"
cd "$REPO_DIR" || exit 1

echo "Experiment: $EXPERIMENT | task ${SLURM_ARRAY_TASK_ID:-1} | node $(hostname) | $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

"$VENV_PYTHON" scripts/pipeline/populate_pose.py "$EXPERIMENT" \
    --processes "$PROCESSES" \
    --batch-size "$BATCH_SIZE" \
    --time-budget-h "$TIME_BUDGET_H" \
    --max-wait 60

EXIT_CODE=$?
echo "Finished: $(date) | exit $EXIT_CODE"
exit $EXIT_CODE
