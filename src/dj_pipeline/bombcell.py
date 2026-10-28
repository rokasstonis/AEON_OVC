"""Bombcell unit labels (aeon_foragingOVC_ovc_bombcell).

Labels every unit in spike_sorting.SortedSpikes.Unit with SpikeInterface's
bombcell_label_units, computed from the quality + template metrics already
stored in spike_sorting.SortingQuality.Metric (no analyzer is loaded). The
labels sit beside the pipeline: they do not change SortedSpikes.unit_quality,
curation or unit matching. Use src.dj_pipeline.unit_spikes to fetch spikes
filtered by these labels.
"""

import json

import datajoint as dj
import pandas as pd

from aeon.dj_pipeline import get_schema_name, spike_sorting
from spikeinterface.curation import bombcell_get_default_thresholds, bombcell_label_units

schema = dj.Schema(get_schema_name("ovc_bombcell"))


@schema
class BombcellParamSet(dj.Lookup):
    definition = """
    bombcell_paramset_id: int16
    ---
    thresholds: <blob>             # {"noise": {...}, "mua": {...}, "non-somatic": {...}}, see bombcell_label_units
    label_non_somatic: bool        # label axonal-looking units "non_soma"
    paramset_description='': varchar(1000)
    """
    contents = [
        (1, bombcell_get_default_thresholds(), True, "SpikeInterface 0.104.9 default bombcell thresholds"),
    ]


@schema
class BombcellLabel(dj.Computed):
    definition = """
    -> spike_sorting.SortingQuality
    -> BombcellParamSet
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> spike_sorting.SortedSpikes.Unit
        ---
        bombcell_label: varchar(16)  # good / mua / noise / non_soma
        """

    def make(self, key):
        thresholds, label_non_somatic = (BombcellParamSet & key).fetch1("thresholds", "label_non_somatic")

        # qc_metrics is json; MariaDB returns json columns as strings
        rows = (spike_sorting.SortingQuality.Metric & key).to_dicts()
        metrics = pd.DataFrame(
            [json.loads(r["qc_metrics"]) if isinstance(r["qc_metrics"], str) else r["qc_metrics"] for r in rows],
            index=[r["unit"] for r in rows],
        ).apply(pd.to_numeric, errors="coerce")

        self.insert1(key)
        if metrics.empty:
            return
        labels = bombcell_label_units(
            external_metrics=metrics, thresholds=thresholds, label_non_somatic=bool(label_non_somatic)
        )["bombcell_label"]
        self.Unit.insert([{**key, "unit": int(unit), "bombcell_label": label} for unit, label in labels.items()])
