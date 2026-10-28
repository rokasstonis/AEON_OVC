"""Summarise step 5 (unit matching) for one experiment: units and spikes across all chunks.

Run from the repo root (so datajoint.json is picked up):

    uv run python scripts/pipeline/unit_matching_summary.py ProjectAeonOVC-2026-08-31_001
    uv run python scripts/pipeline/unit_matching_summary.py ProjectAeonOVC-2026-08-31_001 --quality good
    uv run python scripts/pipeline/unit_matching_summary.py ProjectAeonOVC-2026-08-31_001 --units

Unit matching includes every unit not labelled "noise". Without manual curation
the labels are Kilosort's KSLabel (good/mua), so MUA is included; with manual
curation they are the curator's (good/ok/mua/noise). A global unit can carry
different labels in different blocks; it is reported under the label of the
block(s) holding most of its spikes, and --quality keeps only spikes from
blocks where the unit has one of the given labels.

Spike totals come from UnitMatching.Spikes, where each (global unit, chunk) is
owned by one block only, so spikes in block overlaps are counted once. The
SyncedSpikes total counts overlapping chunks once per block, so it is expected
to be larger; it is shown for comparison. Only spike_count columns are read,
no spike-time files.
"""

import argparse
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("experiment_name", help="e.g. ProjectAeonOVC-2026-08-31_001")
parser.add_argument("-q", "--quality", nargs="+", help="Only count these labels, e.g. --quality good ok")
parser.add_argument("--units", action="store_true", help="Also list spike counts per global unit")
args = parser.parse_args()

from aeon.dj_pipeline import spike_sorting as ss  # noqa: E402 (after argparse so --help needs no DB)

r = {"experiment_name": args.experiment_name}
keep = {q.lower() for q in args.quality} if args.quality else None

n_synced = len(ss.SyncedSpikes & r)
n_matched = len(ss.UnitMatching & r)
print(f"Experiment: {args.experiment_name}")
print(f"  Blocks x shanks synced:  {n_synced}")
print(f"  Blocks x shanks matched: {n_matched}" + ("" if n_matched == n_synced else "   <-- step 5 incomplete"))
if not n_matched:
    raise SystemExit("No UnitMatching entries; run step 5 first.")
if keep:
    print(f"  Counting only labels:    {', '.join(sorted(keep))}")

# Per-block unit labels: (subject, insertion, block_start, shank, unit) -> label
label_of = {
    (s, int(i), b, g, int(u)): str(q).lower()
    for s, i, b, g, u, q in zip(
        *(ss.SortedSpikes.Unit & r).to_arrays(
            "subject", "insertion_number", "block_start", "electrode_group", "unit", "unit_quality"
        )
    )
}
# Per-block global unit -> label in that block
gu_label = {}
for s, i, b, g, u, gu in zip(
    *(ss.UnitMatching.Unit & r).to_arrays(
        "subject", "insertion_number", "block_start", "electrode_group", "unit", "global_unit"
    )
):
    gu_label[(s, int(i), b, g, int(gu))] = label_of.get((s, int(i), b, g, int(u)), "?")

print("\nUnits per block by label (all matched units, before --quality):")
print("  " + ", ".join(f"{q}: {n}" for q, n in Counter(gu_label.values()).most_common()))

# (subject, insertion, shank) -> stats; (subject, insertion, global_unit) -> stats
groups = defaultdict(lambda: {"units": set(), "spikes": 0, "chunks": set(), "synced_spikes": 0})
unit_stats = defaultdict(lambda: {"spikes": 0, "chunks": 0, "labels": Counter(), "shank": None})

for s, i, b, g, gu, chunk, count in zip(
    *(ss.UnitMatching.Spikes & r).to_arrays(
        "subject", "insertion_number", "block_start", "electrode_group", "global_unit", "chunk_start", "spike_count"
    )
):
    i, gu, count = int(i), int(gu), int(count)
    label = gu_label.get((s, i, b, g, gu), "?")
    if keep and label not in keep:
        continue
    grp = groups[(s, i, g)]
    grp["units"].add(gu)
    grp["spikes"] += count
    grp["chunks"].add(chunk)
    u = unit_stats[(s, i, gu)]
    u["spikes"] += count
    u["chunks"] += 1
    u["labels"][label] += count
    u["shank"] = g

for s, i, g, count in zip(
    *(ss.SyncedSpikes.Unit & r).to_arrays("subject", "insertion_number", "electrode_group", "spike_count")
):
    if (s, int(i), g) in groups:
        groups[(s, int(i), g)]["synced_spikes"] += int(count)

print(f"\n{'subject':14} {'ins':>3} {'shank':8} {'units':>6} {'chunks':>6} {'spikes':>14} {'synced (w/ overlaps)':>21}")
tot_units = tot_spikes = tot_synced = 0
for (s, i, g), grp in sorted(groups.items()):
    print(
        f"{s:14} {i:>3} {g:8} {len(grp['units']):>6} {len(grp['chunks']):>6} "
        f"{grp['spikes']:>14,} {grp['synced_spikes']:>21,}"
    )
    tot_units += len(grp["units"])
    tot_spikes += grp["spikes"]
    tot_synced += grp["synced_spikes"]
print(f"{'TOTAL':27} {tot_units:>6} {'':>6} {tot_spikes:>14,} {tot_synced:>21,}")
if keep:
    print("(synced column is all units, not filtered by label)")

# Each global unit under the label holding most of its spikes
by_label = defaultdict(lambda: [0, 0])
for u in unit_stats.values():
    label = u["labels"].most_common(1)[0][0]
    by_label[label][0] += 1
    by_label[label][1] += u["spikes"]
print(f"\n{'label':8} {'global units':>12} {'spikes':>14}")
for label, (n_units, n_spikes) in sorted(by_label.items(), key=lambda kv: -kv[1][1]):
    print(f"{label:8} {n_units:>12} {n_spikes:>14,}")

if args.units:
    print(f"\n{'subject':14} {'ins':>3} {'shank':8} {'global_unit':>11} {'chunks':>6} {'spikes':>12}  labels")
    for (s, i, gu), u in sorted(unit_stats.items()):
        labels = " ".join(f"{q}:{n:,}" for q, n in u["labels"].most_common())
        print(f"{s:14} {i:>3} {u['shank']:8} {gu:>11} {u['chunks']:>6} {u['spikes']:>12,}  {labels}")
