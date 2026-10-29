"""Check whether stored ephys sync models carry the 1 s HARP offset.

Each ONIX HarpSync CSV has two HARP columns: `Value.HarpTime` (correct for OpenEphys.Onix1
>= 0.4.0, which all OVC recordings use) and `Seconds` (= HarpTime + 1 s, i.e. 1 s late).
aeon_mecha fitted the ONIX -> HARP model on `Seconds` from PR #592 (2026-06-26) until
PR #611 (2026-09-17); models ingested with that code put every synced spike / IMU sample 1 s
late. EphysSyncModel.sync_start is the first HARP value of the column the model was fitted
on, so comparing it with both columns of the raw CSV shows which code produced each row.

Run from the repo root (so datajoint.json is picked up):

    uv run python scripts/pipeline/check_sync_offset.py ProjectAeonOVC-2026-08-31_001 ProjectAeonOVC-2026-09-14_001
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_names", nargs="+", help="e.g. ProjectAeonOVC-2026-09-14_001")
args = parser.parse_args()

import pandas as pd  # noqa: E402 (after argparse so --help needs no DB)

from aeon.dj_pipeline import acquisition, ephys  # noqa: E402
from aeon.dj_pipeline.utils.ephys_utils import harp_to_naive  # noqa: E402


def csv_starts(raw_dir):
    """{first HARP datetime: column} for both HARP columns of every HarpSync CSV."""
    starts = {}
    for csv in sorted(Path(raw_dir).rglob("*_HarpSync_*.csv")):
        first = pd.read_csv(csv, nrows=1)
        if first.empty:
            continue
        starts[harp_to_naive(first["Value.HarpTime"].iloc[0])] = "Value.HarpTime"
        starts.setdefault(harp_to_naive(first["Seconds"].iloc[0]), "Seconds")
    return starts


for exp in args.experiment_names:
    key = {"experiment_name": exp}
    raw_dir = acquisition.Experiment.get_data_directory(key, directory_type="raw-ephys")
    rows = (ephys.EphysSyncModel & key).to_arrays("sync_start")
    if raw_dir is None or not len(rows):
        print(f"{exp}: no raw-ephys directory or no EphysSyncModel rows")
        continue
    starts = csv_starts(raw_dir)
    found = pd.Series([starts.get(pd.Timestamp(t).to_pydatetime(), "no matching CSV") for t in rows]).value_counts()
    late = int(found.get("Seconds", 0))
    verdict = "1 s LATE - re-ingest sync models" if late else "OK (fitted on Value.HarpTime)"
    print(f"{exp}: {len(rows)} sync models -> {dict(found)}  => {verdict}")
