"""Fetch spikes across blocks (global units) or within blocks, filtered by bombcell label.

Each function returns a DataJoint query, so it can be restricted further before
fetching, e.g. ``.to_dicts()`` or ``pd.DataFrame(q.to_dicts())``.

    from src.dj_pipeline import unit_spikes as us

    exp = "ProjectAeonOVC-2026-08-31_001"

    # Across blocks: one row per (global unit, chunk), each chunk taken from the
    # block that owns it, labelled with the unit's bombcell label in that block
    q = us.global_unit_spikes(exp, labels=["good"])

    # Global units that are good in every block they appear in
    gl = us.global_unit_labels(exp)
    keep = gl[gl["good"] == gl["n_blocks"]][["subject", "insertion_number", "global_unit"]]
    q = us.global_unit_spikes(exp) & keep.to_dict("records")

    # Within blocks: one row per (block, unit, chunk); overlapping chunks appear
    # once per block. matched=False gives units with no global unit.
    q = us.block_unit_spikes(exp, labels=["good", "mua"], matched=False)

Units without a BombcellLabel row (populate_bombcell.py not run yet) are left out.
"""

import pandas as pd

from aeon.dj_pipeline import spike_sorting as ss

from .bombcell import BombcellLabel

GLOBAL_UNIT_KEY = ["subject", "insertion_number", "global_unit"]


def labelled_units(labels=None, bombcell_paramset_id=1):
    """Per-block units with their bombcell label, optionally only the given labels."""
    q = BombcellLabel.Unit & {"bombcell_paramset_id": bombcell_paramset_id}
    if labels:
        q &= [{"bombcell_label": label} for label in labels]
    return q.proj("bombcell_label")


def global_unit_spikes(experiment_name, labels=None, bombcell_paramset_id=1):
    """Spikes of matched (global) units across blocks.

    One row per (global unit, chunk) from UnitMatching.Spikes; overlaps are counted
    once. A row is kept if the unit's label in the owning block is in ``labels``.
    """
    return (
        ss.UnitMatching.Spikes
        * ss.UnitMatching.Unit.proj("global_unit")
        * labelled_units(labels, bombcell_paramset_id)
        & {"experiment_name": experiment_name}
    )


def block_unit_spikes(experiment_name, labels=None, bombcell_paramset_id=1, matched=None):
    """Spikes of per-block units within each block (SyncedSpikes.Unit).

    ``matched``: None for all units, True for units that have a global unit,
    False for units that unit matching skipped.
    """
    q = ss.SyncedSpikes.Unit * labelled_units(labels, bombcell_paramset_id) & {"experiment_name": experiment_name}
    if matched is True:
        q &= ss.UnitMatching.Unit.proj()
    elif matched is False:
        q -= ss.UnitMatching.Unit.proj()
    return q


def global_unit_labels(experiment_name, bombcell_paramset_id=1) -> pd.DataFrame:
    """One row per global unit: n_blocks, a count of blocks per label, and its shank."""
    rows = (
        ss.UnitMatching.Unit.proj("global_unit") * labelled_units(None, bombcell_paramset_id)
        & {"experiment_name": experiment_name}
    ).to_dicts()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[*GLOBAL_UNIT_KEY, "electrode_group", "n_blocks"])
    counts = pd.crosstab([df[k] for k in GLOBAL_UNIT_KEY], df["bombcell_label"])
    counts.columns.name = None
    counts.insert(0, "n_blocks", counts.sum(axis=1))
    shank = df.groupby(GLOBAL_UNIT_KEY)["electrode_group"].first()
    return counts.join(shank).reset_index()
