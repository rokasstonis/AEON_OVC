#!/bin/bash -l
# =============================================================
# submit_sleap.sh — SLURM array job for SLEAP inference
#                   SWC HPC, gpu_branco partition
#
# Submits the inference array AND a concat job that waits for
# all array tasks to finish before stitching predictions together.
#
# Usage:
#   1. Generate video list:
#        find /path/to/videos -name "*.mp4" | sort > video_list.txt
#   2. Set N below to: wc -l video_list.txt
#   3. sbatch scripts/hpc/submit_sleap.sh
# =============================================================

#SBATCH --job-name=sleap_inference
#SBATCH --partition=gpu_branco
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-29%5            # ← update to wc -l video_list.txt
#SBATCH --output=/ceph/branco/Rokas/Ephys/Results/Behaviour/logs/sleap_%A_%a.out
#SBATCH --error=/ceph/branco/Rokas/Ephys/Results/Behaviour/logs/sleap_%A_%a.err

# =============================================================
# USER SETTINGS
# =============================================================

SESSION_ID="2025-03-26_001"
VIDEO_LIST="/ceph/branco/Rokas/Ephys/Results/Behaviour/${SESSION_ID}/video_list.txt"
CENTROID_MODEL="/ceph/branco/Rokas/Ephys/SLEAP/260326/models/260501_113825.centroid.n=13"
RESULTS_DIR="/ceph/branco/Rokas/Ephys/Results/Behaviour/${SESSION_ID}/sleap_predictions"
OUTPUT_FORMAT="h5"
REPO_DIR="/ceph/branco/Rokas/Ephys/Pipeline"
CONDA_BASE="/ceph/apps/ubuntu-20/packages/miniconda/23.10.0"
CONDA_ENV="sleap_legacy"

# =============================================================
# On the very first task (task 1), submit the concat job as a
# dependency so it waits for the whole array to finish.
# afterok:ARRAY_JOB_ID means all tasks must exit 0.
# =============================================================

if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    echo "Task 1: submitting concat job with dependency on array ${SLURM_ARRAY_JOB_ID}"
    sbatch \
        --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --export=NONE \
        "${REPO_DIR}/scripts/hpc/submit_sleap_concat.sh" \
        "${VIDEO_LIST}" \
        "${RESULTS_DIR}" \
        "${SESSION_ID}"
    echo "Concat job submitted — will start automatically when all inference tasks complete"
fi

# =============================================================
# Environment
# =============================================================

mkdir -p "${RESULTS_DIR}"
mkdir -p "/ceph/branco/Rokas/Ephys/Results/Behaviour/logs"

echo "============================================"
echo "Array job : ${SLURM_ARRAY_JOB_ID}"
echo "Task      : ${SLURM_ARRAY_TASK_ID}"
echo "Node      : $(hostname)"
echo "Started   : $(date)"
echo "============================================"

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs visible: {len(gpus)}')
if not gpus: exit(1)
" || exit 1

# =============================================================
# Get this task's video
# =============================================================

VIDEO_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${VIDEO_LIST}")

if [ -z "${VIDEO_PATH}" ]; then
    echo "ERROR: no video for task ${SLURM_ARRAY_TASK_ID}"; exit 1
fi
if [ ! -f "${VIDEO_PATH}" ]; then
    echo "ERROR: file not found: ${VIDEO_PATH}"; exit 1
fi

VIDEO_NAME=$(basename "${VIDEO_PATH}" | sed 's/\.[^.]*$//')
OUTPUT_FILE="${RESULTS_DIR}/${VIDEO_NAME}.${OUTPUT_FORMAT}"

echo "Video  : ${VIDEO_PATH}"
echo "Output : ${OUTPUT_FILE}"

# Skip if already done — safe to resubmit partial failures
if [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
    echo "Already complete — skipping"
    exit 0
fi

# =============================================================
# Inference
# =============================================================

echo "Starting inference at $(date)"

sleap-track "${VIDEO_PATH}" \
    --model "${CENTROID_MODEL}" \
    --output "${RESULTS_DIR}/${VIDEO_NAME}.slp" \
    --batch-size 8 \
    --peak-threshold 0.2 \
    --verbosity json

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ] && [ "${OUTPUT_FORMAT}" = "h5" ]; then
    sleap-convert \
        "${RESULTS_DIR}/${VIDEO_NAME}.slp" \
        --format analysis \
        --output "${OUTPUT_FILE}"
    [ $? -eq 0 ] && rm -f "${RESULTS_DIR}/${VIDEO_NAME}.slp"
fi

echo "Finished : $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
