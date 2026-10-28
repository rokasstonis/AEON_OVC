"""Label one experiment's sorted units with bombcell and summarise the labels.

Run from the repo root (so datajoint.json is picked up), after step 4
(SortingQuality must exist):

    uv run python scripts/pipeline/populate_bombcell.py ProjectAeonOVC-2026-08-31_001
    uv run python scripts/pipeline/populate_bombcell.py ProjectAeonOVC-2026-08-31_001 --status

Labels go to ovc_bombcell.BombcellLabel.Unit; fetch spikes filtered by them with
src.dj_pipeline.unit_spikes. Only units in SortedSpikes are labelled, i.e. those
carried forward from sorting (Kilosort "good" only while paramset 403 keeps
keep_good_only=True).
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_name", help="e.g. ProjectAeonOVC-2026-08-31_001")
parser.add_argument("-p", "--paramset", type=int, default=1, help="BombcellParamSet id (default: 1)")
parser.add_argument("--status", action="store_true", help="Only print the summary, don't populate")
args = parser.parse_args()

import pandas as pd  # noqa: E402 (after argparse so --help needs no DB)

from aeon.dj_pipeline import spike_sorting as ss  # noqa: E402
from src.dj_pipeline import bombcell  # noqa: E402

r = {"experiment_name": args.experiment_name}
p = {"bombcell_paramset_id": args.paramset}

if not args.status:
    bombcell.BombcellLabel.populate(r, p, display_progress=True)

n_quality = len(ss.SortingQuality & r)
n_labelled = len(bombcell.BombcellLabel & r & p)
print(f"\nBlocks x shanks with SortingQuality: {n_quality}, labelled: {n_labelled}"
      + ("" if n_labelled == n_quality else "   <-- incomplete"))

rows = (ss.SortedSpikes.Unit.proj("unit_quality") * bombcell.BombcellLabel.Unit.proj("bombcell_label") & r & p).to_dicts()
if not rows:
    raise SystemExit("No labels yet.")
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)

print("\nBombcell labels per block and shank:")
per_block = pd.crosstab([df["block_start"], df["electrode_group"]], df["bombcell_label"], margins=True, margins_name="total")
print(per_block.to_string())

print("\nKilosort / curation label (rows) vs bombcell label (columns):")
print(pd.crosstab(df["unit_quality"], df["bombcell_label"], margins=True, margins_name="total").to_string())
