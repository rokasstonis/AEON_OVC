"""Populate pose.SleapRawTracking for one experiment with legacy SLEAP 1.4.

Adapted from submodules/foragingABC_analysis/scripts/populate_sleap_tracking.py.
Run from the repo root (so datajoint.json is picked up) on a GPU node:

    uv run python scripts/pipeline/populate_pose.py ProjectAeonOVC-2026-09-14_001 --max-calls 1
    uv run python scripts/pipeline/populate_pose.py ProjectAeonOVC-2026-09-14_001 --status
    uv run python scripts/pipeline/populate_pose.py ProjectAeonOVC-2026-09-14_001 --downstream --processes 8

Inserts the SleapTrackingMethod/CameraView rows (idempotent), patches
foragingABC to call legacy `sleap-track` (see sleap_legacy.py), then populates.
With reserve_jobs=True, several SLURM tasks can run this concurrently.

Resuming: a task killed mid-chunk leaves its job either `reserved` (hard kill)
or `error` with "SystemExit"/"KeyboardInterrupt" (SIGTERM at the SLURM time
limit, Ctrl-C). populate() skips both, so on startup those job rows are deleted
and populate's queue refresh re-adds them as pending. A reservation counts as
orphaned only if its DB connection is gone, so chunks running in other tasks
are left alone. Real errors (e.g. SleapPipelineError) are kept; retry them with
--retry-errors.
"""

import argparse
import os
import random
import time

SLEAP_METHOD = "ovc_top_260921_legacy"
CAMERA = "CameraTop"
CENTROID_MODEL = "/ceph/branco/Rokas/Ephys/SLEAP/210926/260921_182212.centroid.n=63"
CENTERED_MODEL = "/ceph/branco/Rokas/Ephys/SLEAP/210926/260921_185649.centered_instance.n=63"
INTERRUPTED = ("SystemExit", "KeyboardInterrupt")  # error_message prefixes

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_name", help="e.g. ProjectAeonOVC-2026-09-14_001")
parser.add_argument("-p", "--processes", type=int, default=1, help="Chunks inferred concurrently (default: 1)")
parser.add_argument("-b", "--batch-size", type=int, default=8, help="sleap-track batch size (default: 8)")
parser.add_argument("-n", "--max-calls", type=int, default=None, help="Stop after this many chunks (default: all)")
parser.add_argument("-T", "--time-budget-h", type=float, default=None,
                    help="Don't start a new chunk unless it is expected to finish within this many hours "
                         "of startup (set below the SLURM time limit)")
parser.add_argument("-w", "--max-wait", type=float, default=0,
                    help="Random 0..MAX_WAIT s delay before populate, to stagger SLURM array tasks")
parser.add_argument("--status", action="store_true", help="Print progress and exit")
parser.add_argument("--retry-errors", action="store_true", help="Delete all error jobs first, so they are retried")
parser.add_argument("--downstream", action="store_true",
                    help="Populate PoseQC, PositionPixel, PositionCM, HeadingDirection and "
                         "ovc_behav.EarHeading (CPU only) and exit")
args = parser.parse_args()

# One intra-op thread per job unless the shell/SLURM already set it
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sleap_legacy  # noqa: E402  (same directory)
from foragingabc.dj_pipeline import pose  # noqa: E402

restriction = {"experiment_name": args.experiment_name, "sleap_method": SLEAP_METHOD}
jobs = pose.SleapRawTracking.jobs


def orphaned_reservations():
    """Reserved jobs whose DB connection no longer exists (worker died)."""
    live = {row[0] for row in jobs.connection.query("SELECT ID FROM information_schema.PROCESSLIST")}
    reserved = (jobs.reserved & restriction).to_dicts()
    return [j for j in reserved if j["connection_id"] not in live], len(reserved)


def error_jobs():
    """(interrupted, real) error job rows."""
    errors = (jobs.errors & restriction).to_dicts()
    interrupted = [j for j in errors if j["error_message"].startswith(INTERRUPTED)]
    return interrupted, [j for j in errors if j not in interrupted]


def delete_job(job):
    (jobs & {k: job[k] for k in jobs.primary_key}).delete_quick()


def print_status():
    total = len(pose.SleapRawTracking.key_source & restriction)
    done = len(pose.SleapRawTracking & restriction)
    orphans, n_reserved = orphaned_reservations()
    interrupted, real_errors = error_jobs()
    print(f"{args.experiment_name} [{SLEAP_METHOD}]: {done}/{total} chunks done")
    print(f"  pending : {len(jobs.pending & restriction)}")
    print(f"  reserved: {n_reserved - len(orphans)} running, {len(orphans)} orphaned (reset on next run)")
    print(f"  error   : {len(real_errors)} real, {len(interrupted)} interrupted (reset on next run)")
    for err in real_errors:
        print(f"    {err['chunk_start']}  {err['error_message'][:150]}")


def reset_interrupted():
    """Delete job rows of interrupted chunks so populate re-queues them."""
    orphans, _ = orphaned_reservations()
    for job in orphans:
        print(f"Resetting orphaned reservation: {job['chunk_start']} "
              f"(host {job['host']}, pid {job['pid']}, reserved {job['reserved_time']})", flush=True)
        delete_job(job)
    interrupted, real_errors = error_jobs()
    to_reset = interrupted + (real_errors if args.retry_errors else [])
    for job in to_reset:
        print(f"Resetting error job: {job['chunk_start']}  {job['error_message'][:100]}", flush=True)
        delete_job(job)


if args.status:
    print_status()
    raise SystemExit(0)

if args.downstream:
    # CPU-only tables computed from SleapRawTracking.pose_data; only chunks
    # with a SleapRawTracking row are processed, so re-run after SLEAP finishes.
    from src.dj_pipeline import ovc_behav

    n_sleap = len(pose.SleapRawTracking & restriction)
    for table in (pose.PoseQC, pose.PositionPixel, pose.PositionCM, pose.HeadingDirection,
                  ovc_behav.EarHeading):
        result = table.populate(restriction, display_progress=True, reserve_jobs=True,
                                suppress_errors=True, processes=args.processes)
        print(f"{table.__name__}: {len(table & restriction)}/{n_sleap} chunks, "
              f"{len(result['error_list'])} error(s)", flush=True)
        for key, err in result["error_list"][:3]:
            print(f"  {key.get('chunk_start')}  {str(err)[:200]}", flush=True)
    raise SystemExit(0)

pose.SleapTrackingMethod.insert1(
    {
        "sleap_method": SLEAP_METHOD,
        "multi_animal": 0,
        "centroid_model": CENTROID_MODEL,
        "centered_model": CENTERED_MODEL,
        "description": "OVC CameraTop, SLEAP 1.4 legacy (sleap-track), 7-node skeleton",
    },
    skip_duplicates=True,
)
pose.SleapTrackingMethod.CameraView.insert1(
    {"sleap_method": SLEAP_METHOD, "device_name": CAMERA}, skip_duplicates=True
)

sleap_legacy.apply()

t_start = time.time()
if args.max_wait > 0:
    wait = random.uniform(0, args.max_wait)
    print(f"Waiting {wait:.0f} s before populate", flush=True)
    time.sleep(wait)

reset_interrupted()
print_status()
print(f"processes={args.processes}, batch_size={args.batch_size}, max_calls={args.max_calls}, "
      f"time_budget_h={args.time_budget_h}", flush=True)

# Populate a few chunks at a time, so the task can stop before its time limit
# instead of being killed mid-chunk.
budget_s = args.time_budget_h * 3600 if args.time_budget_h else None
n_done = n_failed = 0
longest_s = 0.0
while args.max_calls is None or n_done + n_failed < args.max_calls:
    if budget_s and (time.time() - t_start) + 1.2 * longest_s > budget_s:
        print(f"Time budget reached (longest chunk {longest_s / 60:.0f} min), stopping", flush=True)
        break
    batch = args.processes if args.max_calls is None else min(args.processes, args.max_calls - n_done - n_failed)
    t_batch = time.time()
    result = pose.SleapRawTracking.populate(
        restriction,
        display_progress=True,
        processes=args.processes,
        reserve_jobs=True,
        suppress_errors=True,
        max_calls=batch,
        make_kwargs={"batch_size": args.batch_size},
    )
    n_ok, n_err = result["success_count"], len(result["error_list"])
    if n_ok + n_err == 0:
        print("No pending chunks left", flush=True)
        break
    longest_s = max(longest_s, time.time() - t_batch)
    n_done, n_failed = n_done + n_ok, n_failed + n_err
    for key, err in result["error_list"]:
        print(f"Error: {key.get('chunk_start')}  {str(err)[:200]}", flush=True)

print(f"Done: {n_done} chunk(s) populated, {n_failed} error(s), "
      f"{(time.time() - t_start) / 3600:.1f} h", flush=True)
