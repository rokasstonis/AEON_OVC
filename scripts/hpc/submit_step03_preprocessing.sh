#!/bin/bash

# =============================================================================
# Step 03 (parts 1-2) as a SLURM CPU job array: sorting prerequisites + PreProcessing
# =============================================================================
# Uses the EXPERIMENT_NAME / PARAMSET_ID / SORTING_GROUPS set at the top of
# submodules/aeon_mecha/docs/ephys_runbooks/step03_spike_sorting.py.
#
# Each array task is a worker that keeps taking pending PreProcessing entries
# (block x shank) one at a time until none are left, so the entries run in
# parallel across workers. Task 1 also runs setup_sorting_prerequisites(); the
# other tasks wait for its SortingTask rows. DataJoint job reservation
# (reserve_jobs=True) makes sure no two workers process the same entry; a worker
# that loses a reservation race just moves on to the next entry. Already
# preprocessed entries are skipped, so resubmitting only runs what's missing.
# Preprocessing 24 h blocks writes ~0.5 TB per shank, so never run it on the
# login node (it gets killed silently).
#
# --array sets the number of parallel workers (6 keeps the load on the shared
# Ceph storage reasonable); it does not need to match the number of entries.
#
# Before submitting (from the AEON_OVC root): module load uv && uv sync && mkdir -p slurm_output
# (the venv is synced once up front, not by every task in parallel.)
#
# Usage:  sbatch scripts/hpc/submit_step03_preprocessing.sh
#
# Afterwards, generate the GPU sorting scripts on the login node (DB-only):
#   uv run python submodules/aeon_mecha/docs/ephys_runbooks/step03_spike_sorting.py
# (setup and preprocessing are skipped when already done).
#
# Job status lives in spike_sorting.PreProcessing.jobs (.progress(), .reserved,
# .errors). If a task is killed, its entry stays "reserved"; clear it with
#   spike_sorting.PreProcessing.jobs.reserved.delete()
# (only when no workers are running) and delete that entry's partial
# recording.zarr on /ceph/scratch before resubmitting.
# =============================================================================

#SBATCH --job-name=aeon-preprocessing
#SBATCH --partition=cpu_branco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                    # fork_safe_job_kwargs caps SI workers at 8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=1-12                          # 6 parallel workers draining the entries
#SBATCH --output=slurm_output/preproc_%A_%a.out
#SBATCH --error=slurm_output/preproc_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rokas.stonis.19@ucl.ac.uk

set -e

TASK=${SLURM_ARRAY_TASK_ID:-1}
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}  Task: $TASK  Node: $SLURM_NODELIST  Start: $(date)"

cd "$SLURM_SUBMIT_DIR"
if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv not found in $(pwd) - run 'uv sync' before submitting"
    exit 1
fi

# OpenBLAS/MKL otherwise spawn one thread per core on the node, which can hit
# the per-user thread limit (pthread_create error 11).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TASK

.venv/bin/python - <<'EOF'
import os
import runpy
import time

step03 = runpy.run_path("submodules/aeon_mecha/docs/ephys_runbooks/step03_spike_sorting.py")
experiment_name = step03["EXPERIMENT_NAME"]
paramset_id = str(step03["PARAMSET_ID"])
task = int(os.environ["TASK"])

from aeon.dj_pipeline import spike_sorting

restriction = {"experiment_name": experiment_name, "paramset_id": paramset_id}

if task == 1:
    print("\n--- Setup sorting prerequisites ---")
    step03["setup_sorting_prerequisites"](
        experiment_name,
        step03["PARAMSET_ID"],
        step03["SORTING_METHOD"],
        insertion_number=step03["INSERTION_NUMBER"],
        sorting_groups=step03["SORTING_GROUPS"],
    )
else:
    # Wait for task 1 to insert the SortingTask rows (poll every 30 s, up to 30 min).
    for _ in range(60):
        if spike_sorting.SortingTask & restriction:
            break
        time.sleep(30)
    else:
        raise RuntimeError(f"No SortingTask rows for {restriction} after 30 min - did task 1 fail?")

# Take one pending entry at a time until none are left. Re-fetching after each
# entry (rather than populate's single up-front key list) means a worker that
# loses a reservation race to another worker just moves on to the next entry.
print(f"\n--- Task {task}: PreProcessing worker ---")
errors = []
idle_rounds = 0
while True:
    result = spike_sorting.PreProcessing.populate(
        restriction,
        reserve_jobs=True,
        max_calls=1,
        display_progress=True,
        suppress_errors=True,
    )
    errors.extend(result["error_list"])
    attempted = result["success_count"] + len(result["error_list"])
    if not (spike_sorting.PreProcessing.jobs.pending & restriction):
        break
    # Pending entries left but nothing attempted (lost a race, or not yet
    # schedulable): retry shortly, but don't spin forever.
    idle_rounds = 0 if attempted else idle_rounds + 1
    if idle_rounds >= 20:
        print("Pending entries remain but none could be reserved; stopping this worker.")
        break
    if not attempted:
        time.sleep(15)

n_done = len(spike_sorting.PreProcessing & restriction)
n_total = len(spike_sorting.SortingTask & restriction)
print(f"PreProcessing entries done: {n_done}/{n_total}")
print(f"Job status: {spike_sorting.PreProcessing.jobs.progress()}")
for key, err in errors:
    print(f"ERROR in {key}:\n  {err}")
if errors:
    raise SystemExit(1)
EOF

echo "Finished: $(date)"
