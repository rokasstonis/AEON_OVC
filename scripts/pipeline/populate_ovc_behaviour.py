"""Populate the OVC behaviour tables for one experiment: objects, sound stimuli, kinematics, escape trials.

Run from the repo root (so datajoint.json is picked up), CPU only, after
`populate_pose.py <exp> --downstream` (Kinematics needs pose.PositionCM):

    uv run python scripts/pipeline/populate_ovc_behaviour.py ProjectAeonOVC-2026-09-14_001
    uv run python scripts/pipeline/populate_ovc_behaviour.py ProjectAeonOVC-2026-09-14_001 --processes 8
    uv run python scripts/pipeline/populate_ovc_behaviour.py ProjectAeonOVC-2026-09-14_001 --status

Steps (each skips what is already done):
  1. ObjectRegistration  <- <registration root>/<exp>/combined_registration.json (or --registration)
  2. SoundStimulus       <- streams.SoundCardSoundEvents + the epoch's sound bank
  3. Kinematics          <- pose (one row per CameraTop chunk)
  4. ObjectKinematics    <- Kinematics x objects present in that chunk
  5. EscapeTrial         <- one row per sound stimulus (stats only, no escape label)

After changing a registration file, use --replace-registration; that deletes the old
objects and everything computed from them (ObjectKinematics, EscapeTrial) first.
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_name", help="e.g. ProjectAeonOVC-2026-09-14_001")
parser.add_argument("--registration", help="Registration JSON (default: registration root/<exp>/combined_registration.json)")
parser.add_argument("--replace-registration", action="store_true", help="Re-ingest the registration file")
parser.add_argument("-p", "--processes", type=int, default=1, help="Parallel processes for populate (default: 1)")
parser.add_argument("--status", action="store_true", help="Only print counts")
args = parser.parse_args()

from pathlib import Path  # noqa: E402 (after argparse so --help needs no DB)

from foragingabc.dj_pipeline import pose  # noqa: E402
from src.dj_pipeline import ovc_behav, registration  # noqa: E402

r = {"experiment_name": args.experiment_name}
TABLES = (ovc_behav.SoundStimulus, ovc_behav.Kinematics, ovc_behav.ObjectKinematics, ovc_behav.EscapeTrial)


def print_status():
    n_objects = len(ovc_behav.ObjectRegistration.Object & r)
    print(f"\nObjectRegistration.Object: {n_objects} objects")
    print(f"pose.PositionCM:           {len(pose.PositionCM & r)} chunks")
    print(f"SoundStimulus:             {len(ovc_behav.SoundStimulus & r)} chunks, "
          f"{len(ovc_behav.SoundStimulus.Event & r)} events")
    print(f"Kinematics:                {len(ovc_behav.Kinematics & r)} chunks")
    print(f"ObjectKinematics:          {len(ovc_behav.ObjectKinematics & r)} (chunk, object) rows "
          f"of {len(ovc_behav.ObjectKinematics().key_source & r)}")
    print(f"EscapeTrial:               {len(ovc_behav.EscapeTrial & r)} trials "
          f"of {len(ovc_behav.EscapeTrial().key_source & r)} stimuli with kinematics")


if not args.status:
    path = Path(args.registration) if args.registration else (
        registration.REGISTRATION_ROOT / args.experiment_name / registration.REGISTRATION_FILE
    )
    if path.exists():
        registration.ingest(args.experiment_name, path, replace=args.replace_registration)
    else:
        print(f"No registration file at {path}; object tables stay empty.")

    for table in TABLES:
        result = table.populate(r, display_progress=True, reserve_jobs=True,
                                suppress_errors=True, processes=args.processes)
        print(f"{table.__name__}: {len(result['error_list'])} error(s)", flush=True)
        for key, err in result["error_list"][:3]:
            print(f"  {key.get('chunk_start')} {key.get('stim_onset', '')}  {str(err)[:200]}", flush=True)

print_status()
