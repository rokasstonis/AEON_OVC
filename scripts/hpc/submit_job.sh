#!/bin/bash -l
# =============================================================
# submit_job.sh — SWC HPC (SLURM) job script for NP2.0 pipeline
#
# Cluster specifics:
#   Scheduler  : SLURM
#   Lab partition: gpu_branco (30-day limit, L40S and H100 GPUs)
#   Storage    : /ceph/branco/$USER  (lab Ceph — used for both scratch
#                and results since no dedicated /scratch filesystem)
#   CUDA       : 11.8 (matches cu118 PyTorch wheel)
#
# Usage (from login node hpc-gw2):
#   1. Edit USER SETTINGS below
#   2. sbatch scripts/hpc/submit_job.sh
#   3. Monitor: squeue -u $USER
#               tail -f /ceph/branco/$USER/np2_scratch/SESSION/pipeline.log
# =============================================================

# ---- SLURM directives ----
#SBATCH --job-name=np2_pipeline
#SBATCH --partition=gpu_branco       # lab partition — 30-day limit
#SBATCH --gres=gpu:l40s:1           # L40S: 48 GB VRAM, plenty for KS4
                                     # swap to gpu:h100:1 if L40S busy
#SBATCH --cpus-per-task=16          # 16 CPU threads for SpikeInterface
#SBATCH --mem=64G                    # 64 GB RAM
#SBATCH --time=48:00:00             # 48 hr — preprocessing + sorting + postproc
#SBATCH --output=logs/np2_%j.out
#SBATCH --error=logs/np2_%j.err

# =============================================================
# USER SETTINGS — edit these for each session
# =============================================================

SESSION_ID="2025-08-31_001"

# Where your raw .bin and Clock files live on Ceph
CEPH_RAW="/ceph/aeon/data/raw/AEONX1/ProjectAeonOVC/2026-08-31T160655Z/NeuropixelsV2"

# Where results (sorting_analyzer, QC metrics, log) will be written
CEPH_RESULTS="/ceph/branco/Rokas/Ephys/Results/${SESSION_ID}" #/ceph/branco/Rokas/Ephys

# Scratch for large intermediates (preprocessed binary ~4TB, KS output ~5GB)
# On SWC HPC there is no dedicated /scratch — we use Ceph directly.``
# Ceph can handle the sequential writes for preprocessing; KS4 is fine too
# since its output is small. The preprocessed binary stays here after the job
# so you can re-sort without re-preprocessing if needed.
SCRATCH_BASE="/ceph/branco/Rokas/Ephys/Results/np2_scratch/${SESSION_ID}"

CONDA_ENV="np2_pipeline"
CONDA_BASE="/ceph/apps/ubuntu-20/packages/miniconda/23.10.0"
REPO_DIR="/ceph/branco/Rokas/Ephys/Pipeline"

# =============================================================
# Environment setup
# =============================================================

echo "============================================"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Session   : ${SESSION_ID}"
echo "Started   : $(date)"
echo "Scratch   : ${SCRATCH_BASE}"
echo "============================================"

module load cuda/11.8             # matches cu118 PyTorch wheel

# Fail early on path mistakes before wasting allocation time.
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi
if [[ ! -f "${REPO_DIR}/config.yml" ]]; then
    echo "ERROR: config.yml not found at ${REPO_DIR}/config.yml"
    exit 1
fi
if [[ ! -f "${REPO_DIR}/scripts/hpc_pipeline.py" ]]; then
    echo "ERROR: hpc_pipeline.py not found at ${REPO_DIR}/scripts/hpc_pipeline.py"
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Python    : $(which python)"
echo "SI version: $(python -c 'import spikeinterface; print(spikeinterface.__version__)')"
echo "CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Fail immediately if GPU is not visible — don't waste 48hr of allocation
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available — check module load and PyTorch build')
    exit(1)
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu}  ({mem:.1f} GB VRAM)')
" || exit 1

# =============================================================
# Scratch setup and space check
# =============================================================

mkdir -p "${SCRATCH_BASE}"
mkdir -p "logs"

echo ""
echo "Ceph free : $(df -h /ceph/branco | tail -1 | awk '{print $4}')"

# Warn if less than 6 TB free — a 24hr recording needs ~6TB during sorting
python - << EOF
import shutil
free = shutil.disk_usage("/ceph/branco").free / 1e12
print(f"Ceph /ceph/branco free: {free:.1f} TB")
if free < 6:
    print(f"WARNING: only {free:.1f} TB free — a 24hr recording needs ~6 TB.")
    print("Free up space or the job will fail mid-preprocessing.")
    # Don't abort — user may have checked this and knows what they're doing
EOF

# =============================================================
# Build runtime config (patches paths without editing config.yaml)
# =============================================================

python - << EOF
import yaml

with open("${REPO_DIR}/config.yml") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["raw_dir"]      = "${CEPH_RAW}"
cfg["session"]["session_id"] = "${SESSION_ID}"
cfg["sorting"]["n_jobs"]     = int("${SLURM_CPUS_PER_TASK}")

with open("${SCRATCH_BASE}/config_runtime.yml", "w") as f:
    yaml.dump(cfg, f)

print(f"Runtime config written:")
print(f"  raw_dir : ${CEPH_RAW}")
print(f"  n_jobs  : ${SLURM_CPUS_PER_TASK}")
EOF

# =============================================================
# GPU info
# =============================================================

echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# =============================================================
# Run pipeline
# =============================================================

echo ""
echo "Starting hpc_pipeline.py at $(date)"

python "${REPO_DIR}/scripts/hpc_pipeline.py" \
    --config  "${SCRATCH_BASE}/config_runtime.yml" \
    --scratch "${SCRATCH_BASE}" \
    --ceph-results "${CEPH_RESULTS}" \
    --n-jobs  "${SLURM_CPUS_PER_TASK}"

EXIT_CODE=$?

# =============================================================
# Always copy log to results, even on failure
# =============================================================

mkdir -p "${CEPH_RESULTS}"
cp "${SCRATCH_BASE}/pipeline.log" \
   "${CEPH_RESULTS}/pipeline_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "PIPELINE FAILED — inspect:"
    echo "  ${CEPH_RESULTS}/pipeline_${SLURM_JOB_ID}.log"
    echo ""
    echo "If preprocessing completed before failure, the binary is at:"
    echo "  ${SCRATCH_BASE}/processed/preprocessed_binary/"
    echo "Re-submit and it will skip straight to sorting."
fi

exit ${EXIT_CODE}
