"""Fetch spikes across blocks (global units) or within blocks, filtered by bombcell label.

Labels are per block (BombcellLabel.Unit). For filtering, a matched unit's label is its *effective* label:
"good" in every block of a global unit that bombcell labels good in any of its blocks, otherwise the block's own
label. Units without a global unit keep their own label. The stored labels are not changed.

Each spike function returns a DataJoint query, so it can be restricted further before
fetching, e.g. ``.to_dicts()`` or ``pd.DataFrame(q.to_dicts())``.

    from src.dj_pipeline import unit_spikes as us

    exp = "ProjectAeonOVC-2026-08-31_001"

    # Across blocks: one row per (global unit, chunk), each chunk taken from the
    # block that owns it; good global units only
    q = us.global_unit_spikes(exp, labels=["good"])

    # One row per global unit: blocks per raw label and its global_label
    gl = us.global_unit_labels(exp)

    # Within blocks: one row per (block, unit, chunk); overlapping chunks appear
    # once per block. matched=False gives units with no global unit.
    q = us.block_unit_spikes(exp, labels=["good", "mua"], matched=False)

Units without a BombcellLabel row (populate_bombcell.py not run yet) are left out when filtering by label.
"""

import pandas as pd

from aeon.dj_pipeline import spike_sorting as ss

from .bombcell import BombcellLabel

GOOD = "good"
GLOBAL_UNIT_KEY = ["subject", "insertion_number", "global_unit"]
BLOCK_UNIT_KEY = ["subject", "insertion_number", "block_start", "electrode_group", "unit"]


def labelled_units(labels=None, bombcell_paramset_id=1):
    """Per-block units with their raw bombcell label, optionally only the given labels."""
    q = BombcellLabel.Unit & {"bombcell_paramset_id": bombcell_paramset_id}
    if labels:
        q &= [{"bombcell_label": label} for label in labels]
    return q.proj("bombcell_label")


def effective_labels(df, label="bombcell_label"):
    """'good' for every row of a global unit with a good row, else the row's own label (rows without a
    global_unit keep theirs). df: one row per (block, unit) with GLOBAL_UNIT_KEY and `label`."""
    matched = df["global_unit"].notna()
    good = df[matched].groupby(GLOBAL_UNIT_KEY)[label].transform(lambda s: (s == GOOD).any())
    out = df[label].copy()
    out[good[good].index] = GOOD
    return out


def unit_labels(experiment_name, bombcell_paramset_id=1) -> pd.DataFrame:
    """One row per labelled (block, unit): BLOCK_UNIT_KEY, global_unit (NaN if unmatched), bombcell_label (raw)
    and effective_label (see module docstring)."""
    r = {"experiment_name": experiment_name}
    lab = pd.DataFrame((labelled_units(None, bombcell_paramset_id) & r).to_dicts())
    if lab.empty:
        return pd.DataFrame(columns=[*BLOCK_UNIT_KEY, "global_unit", "bombcell_label", "effective_label"])
    gu = pd.DataFrame((ss.UnitMatching.Unit & r).proj("global_unit").to_dicts())
    if gu.empty:
        lab["global_unit"] = pd.NA
    else:
        lab = lab.merge(gu[[*BLOCK_UNIT_KEY, "global_unit"]].drop_duplicates(), on=BLOCK_UNIT_KEY, how="left")
    lab["effective_label"] = effective_labels(lab)
    return lab


def global_unit_spikes(experiment_name, labels=None, bombcell_paramset_id=1):
    """Spikes of matched (global) units across blocks.

    One row per (global unit, chunk) from UnitMatching.Spikes; overlaps are counted
    once. With ``labels``, a row is kept if the unit's effective label in the owning block
    is one of them (for ["good"]: every chunk of a global unit that is good in any block).
    """
    q = ss.UnitMatching.Spikes & {"experiment_name": experiment_name}
    if not labels:
        return q
    lab = unit_labels(experiment_name, bombcell_paramset_id)
    keep = lab[lab["global_unit"].notna() & lab["effective_label"].isin(labels)]
    keep = keep[["subject", "insertion_number", "block_start", "electrode_group", "global_unit"]].astype(
        {"global_unit": int}
    )
    return q & keep.to_dict("records")


def block_unit_spikes(experiment_name, labels=None, bombcell_paramset_id=1, matched=None):
    """Spikes of per-block units within each block (SyncedSpikes.Unit), filtered by effective label.

    ``matched``: None for all units, True for units that have a global unit,
    False for units that unit matching skipped.
    """
    q = ss.SyncedSpikes.Unit & {"experiment_name": experiment_name}
    if labels:
        lab = unit_labels(experiment_name, bombcell_paramset_id)
        q &= lab.loc[lab["effective_label"].isin(labels), BLOCK_UNIT_KEY].to_dict("records")
    if matched is True:
        q &= ss.UnitMatching.Unit.proj()
    elif matched is False:
        q -= ss.UnitMatching.Unit.proj()
    return q


def global_unit_labels(experiment_name, bombcell_paramset_id=1) -> pd.DataFrame:
    """One row per global unit: n_blocks, a count of blocks per raw label, its shank, and global_label
    ('good' if good in any block, else its most common label)."""
    lab = unit_labels(experiment_name, bombcell_paramset_id)
    df = lab[lab["global_unit"].notna()].astype({"global_unit": int})
    if df.empty:
        return pd.DataFrame(columns=[*GLOBAL_UNIT_KEY, "electrode_group", "n_blocks", "global_label"])
    counts = pd.crosstab([df[k] for k in GLOBAL_UNIT_KEY], df["bombcell_label"])
    counts.columns.name = None
    counts.insert(0, "n_blocks", counts.sum(axis=1))
    shank = df.groupby(GLOBAL_UNIT_KEY)["electrode_group"].first()
    out = counts.join(shank).reset_index()
    labels = [c for c in counts.columns if c != "n_blocks"]
    out["global_label"] = out[labels].idxmax(axis=1)
    if GOOD in labels:
        out.loc[out[GOOD] > 0, "global_label"] = GOOD
    return out
