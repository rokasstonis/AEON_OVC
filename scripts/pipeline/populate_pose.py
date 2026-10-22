"""Populate pose.SleapRawTracking for one experiment with legacy SLEAP 1.4.

Adapted from submodules/foragingABC_analysis/scripts/populate_sleap_tracking.py.
Run from the repo root (so datajoint.json is picked up) on a GPU node:

    uv run python scripts/pipeline/populate_pose.py ProjectAeonOVC-2026-09-14_001 --max-calls 1

Inserts the SleapTrackingMethod/CameraView rows (idempotent), patches
foragingABC to call legacy `sleap-track` (see sleap_legacy.py), then populates.
With reserve_jobs=True, several SLURM tasks can run this concurrently.
"""

import argparse
import os
import random
import time

SLEAP_METHOD = "ovc_top_260921_legacy"
CAMERA = "CameraTop"
CENTROID_MODEL = "/ceph/branco/Rokas/Ephys/SLEAP/210926/260921_182212.centroid.n=63"
CENTERED_MODEL = "/ceph/branco/Rokas/Ephys/SLEAP/210926/260921_185649.centered_instance.n=63"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_name", help="e.g. ProjectAeonOVC-2026-09-14_001")
parser.add_argument("-p", "--processes", type=int, default=1, help="Chunks inferred concurrently (default: 1)")
parser.add_argument("-b", "--batch-size", type=int, default=8, help="sleap-track batch size (default: 8)")
parser.add_argument("-n", "--max-calls", type=int, default=None, help="Stop after this many chunks (default: all)")
parser.add_argument("-w", "--max-wait", type=float, default=0,
                    help="Random 0..MAX_WAIT s delay before populate, to stagger SLURM array tasks")
args = parser.parse_args()

# One intra-op thread per job unless the shell/SLURM already set it
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import sleap_legacy  # noqa: E402  (same directory)
from foragingabc.dj_pipeline import pose  # noqa: E402

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

if args.max_wait > 0:
    wait = random.uniform(0, args.max_wait)
    print(f"Waiting {wait:.0f} s before populate", flush=True)
    time.sleep(wait)

restriction = {"experiment_name": args.experiment_name, "sleap_method": SLEAP_METHOD}
todo = len(pose.SleapRawTracking.key_source & restriction) - len(pose.SleapRawTracking & restriction)
print(f"Populating SleapRawTracking for {args.experiment_name}: {todo} chunk(s) pending, "
      f"processes={args.processes}, batch_size={args.batch_size}, max_calls={args.max_calls}", flush=True)

result = pose.SleapRawTracking.populate(
    restriction,
    display_progress=True,
    processes=args.processes,
    reserve_jobs=True,
    max_calls=args.max_calls,
    make_kwargs={"batch_size": args.batch_size},
)
print(f"Done: {result}", flush=True)
