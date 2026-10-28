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

# ---- Unit matching: a unit is "matched" if its global unit spans 2+ blocks ----
gu = pd.DataFrame((ss.UnitMatching.Unit.proj("global_unit") & r).to_dicts())
if gu.empty:
    raise SystemExit("\nNo UnitMatching entries yet; run step 5 for the matching summary.")
gu_key = ["subject", "insertion_number", "global_unit"]
gu["n_blocks"] = gu.groupby(gu_key)["block_start"].transform("nunique")
unit_key = ["subject", "insertion_number", "block_start", "electrode_group", "unit"]
df = df.merge(gu[[*unit_key, "global_unit", "n_blocks"]], on=unit_key, how="left")
df["matching"] = "no global unit"
df.loc[df["n_blocks"] == 1, "matching"] = "this block only"
df.loc[df["n_blocks"] > 1, "matching"] = "matched (2+ blocks)"

print("\nUnits matched across blocks (global unit spans 2+ blocks), per block and shank:")
matched = df[df["n_blocks"] > 1]
per_block_matched = pd.crosstab(
    [matched["block_start"], matched["electrode_group"]], matched["bombcell_label"], margins=True, margins_name="total"
).reindex(columns=per_block.columns, fill_value=0)
print(per_block_matched.to_string())

print("\nMatching status (rows) vs bombcell label (columns), per-block units:")
print(pd.crosstab(df["matching"], df["bombcell_label"], margins=True, margins_name="total").to_string())

# One row per global unit: blocks spanned, and how consistently it is labelled good
g = df.dropna(subset=["global_unit"]).groupby(gu_key).agg(
    n_blocks=("n_blocks", "first"),
    n_good=("bombcell_label", lambda s: int((s == "good").sum())),
)
g["blocks spanned"] = g["n_blocks"].clip(upper=6).astype(int).astype(str).replace("6", "6+")
summary = g.groupby("blocks spanned").agg(
    global_units=("n_blocks", "size"),
    good_in_all_blocks=("n_good", lambda s: int((s == g.loc[s.index, "n_blocks"]).sum())),
    good_in_any_block=("n_good", lambda s: int((s > 0).sum())),
)
summary.loc["total"] = summary.sum()
print("\nGlobal units by number of blocks spanned:")
print(summary.to_string())
