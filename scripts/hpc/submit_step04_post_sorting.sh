#!/bin/bash

# =============================================================================
# Step 04 as a SLURM CPU job array: post-sorting tables + auto-approval curation
# =============================================================================
# Uses the EXPERIMENT_NAME set at the top of
# submodules/aeon_mecha/docs/ephys_runbooks/step04_post_sorting_and_curation.py.
#
# Each array task is a worker that drains the post-sorting tables in dependency
# order -- PostProcessing, SortedSpikes, Waveform, SortingQuality -- taking one
# pending entry (block x shank) at a time, so entries run in parallel across
# workers. DataJoint job reservation (reserve_jobs=True) makes sure no two
# workers process the same entry. A worker that finishes PostProcessing early
# moves on to the downstream tables for entries that are already post-processed;
# whichever worker finishes last fills in the rest. Once every table is complete,
# the worker that notices runs verify_sorting() and auto_approve_curation()
# (both idempotent). Already populated entries are skipped, so resubmitting only
# runs what's missing.
#
# Before submitting (from the AEON_OVC root): module load uv && uv sync && mkdir -p slurm_output
# (the venv is synced once up front, not by every task in parallel.)
#
# Usage:  sbatch scripts/hpc/submit_step04_post_sorting.sh
#
# Job status lives in spike_sorting.<Table>.jobs (.progress(), .reserved,
# .errors). If a task is killed, its entry stays "reserved"; clear it with
#   spike_sorting.PostProcessing.jobs.reserved.delete()
# (only when no workers are running) before resubmitting.
# =============================================================================

#SBATCH --job-name=aeon-postsorting
#SBATCH --partition=cpu_branco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                    # fork_safe_job_kwargs caps SI workers at 8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-6                          # 6 parallel workers draining the entries
#SBATCH --output=slurm_output/postsort_%A_%a.out
#SBATCH --error=slurm_output/postsort_%A_%a.err
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

step04 = runpy.run_path("submodules/aeon_mecha/docs/ephys_runbooks/step04_post_sorting_and_curation.py")
experiment_name = step04["EXPERIMENT_NAME"]
task = int(os.environ["TASK"])

from aeon.dj_pipeline import spike_sorting

restriction = {"experiment_name": experiment_name}

n_sorted = len(spike_sorting.SpikeSorting & restriction)
if n_sorted == 0:
    raise SystemExit("No SpikeSorting entries found. Has the SLURM sorting job completed successfully?")
print(f"SpikeSorting entries: {n_sorted}")

tables = [
    spike_sorting.PostProcessing,
    spike_sorting.SortedSpikes,
    spike_sorting.Waveform,
    spike_sorting.SortingQuality,
]


def drain(table):
    """Take one pending entry at a time until none are left.

    Re-fetching after each entry (rather than populate's single up-front key
    list) means a worker that loses a reservation race to another worker just
    moves on to the next entry.
    """
    errors = []
    idle_rounds = 0
    while True:
        result = table.populate(
            restriction,
            reserve_jobs=True,
            max_calls=1,
            display_progress=True,
            suppress_errors=True,
        )
        errors.extend(result["error_list"])
        attempted = result["success_count"] + len(result["error_list"])
        if not (table.jobs.pending & restriction):
            break
        # Pending entries left but nothing attempted (lost a race, or not yet
        # schedulable): retry shortly, but don't spin forever.
        idle_rounds = 0 if attempted else idle_rounds + 1
        if idle_rounds >= 20:
            print("Pending entries remain but none could be reserved; moving on.")
            break
        if not attempted:
            time.sleep(15)
    return errors


errors = []
for table in tables:
    print(f"\n--- Task {task}: {table.__name__} worker ---")
    errors.extend(drain(table))

counts = {table.__name__: len(table & restriction) for table in tables}
print("\nEntries done:")
for name, count in counts.items():
    print(f"  {name}: {count}/{n_sorted}")

if errors:
    print("\nSkipping curation: errors occurred in this worker.")
elif all(count == n_sorted for count in counts.values()):
    print("\n--- Verify sorting results ---")
    step04["verify_sorting"](experiment_name)
    print("\n--- Auto-approve curation ---")
    step04["auto_approve_curation"](experiment_name)
else:
    print("\nSkipping curation: other workers are still populating (the last one to finish runs it).")

for key, err in errors:
    print(f"ERROR in {key}:\n  {err}")
if errors:
    raise SystemExit(1)
EOF

echo "Finished: $(date)"
