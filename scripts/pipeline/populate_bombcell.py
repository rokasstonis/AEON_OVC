"""Label one experiment's sorted units with bombcell and summarise the labels.

Run from the repo root (so datajoint.json is picked up), after step 4
(SortingQuality must exist):

    uv run python scripts/pipeline/populate_bombcell.py ProjectAeonOVC-2026-08-31_001
    uv run python scripts/pipeline/populate_bombcell.py ProjectAeonOVC-2026-08-31_001 --status

Labels go to ovc_bombcell.BombcellLabel.Unit; fetch spikes filtered by them with
src.dj_pipeline.unit_spikes. Labels are stored per block; the matching tables use
effective labels, where a global unit that is good in any block is good in all of them. Only units in SortedSpikes are labelled, i.e. those
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
from src.dj_pipeline.unit_spikes import GOOD, effective_labels  # noqa: E402

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

# A global unit that is good in any block counts as good in all its blocks (stored labels unchanged)
df["effective_label"] = effective_labels(df)
promoted = df[df["effective_label"] != df["bombcell_label"]]
print(f"\nRelabelled good via their global unit (good in another block): {len(promoted)} per-block units"
      + (f" (were {', '.join(f'{q}: {n}' for q, n in promoted['bombcell_label'].value_counts().items())})"
         if len(promoted) else ""))
print("The tables below use these effective labels.")

print("\nUnits matched across blocks (global unit spans 2+ blocks), per block and shank:")
matched = df[df["n_blocks"] > 1]
per_block_matched = pd.crosstab(
    [matched["block_start"], matched["electrode_group"]], matched["effective_label"], margins=True,
    margins_name="total"
).reindex(columns=per_block.columns, fill_value=0).rename_axis(columns="effective_label")
print(per_block_matched.to_string())

print("\nMatching status (rows) vs effective label (columns), per-block units:")
print(pd.crosstab(df["matching"], df["effective_label"], margins=True, margins_name="total").to_string())

# One row per global unit: blocks spanned and its label (good if good in any block, else its most common label)
g = df.dropna(subset=["global_unit"]).groupby(gu_key).agg(
    n_blocks=("n_blocks", "first"),
    global_label=("bombcell_label", lambda s: GOOD if (s == GOOD).any() else s.mode().iloc[0]),
)
summary = pd.crosstab(g["n_blocks"].astype(int).rename("blocks spanned"), g["global_label"])
summary.columns.name = None
summary = summary[[GOOD, *sorted(c for c in summary.columns if c != GOOD)] if GOOD in summary else list(summary)]
summary.insert(0, "global_units", summary.sum(axis=1))
# Units spanning at least n blocks: sum the exact-span rows from the longest span down
at_least = summary[::-1].cumsum()[::-1]
at_least.index.name = "at least n blocks"
summary.loc["total"] = summary.sum()
print("\nGlobal units by number of blocks spanned (each unit counted once, at its exact span), by global label:")
print(summary.to_string())
print("\nGlobal units spanning at least n blocks (cumulative; n = all blocks = tracked start to end):")
print(at_least.to_string())
