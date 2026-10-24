"""OVC behaviour schema (aeon_foragingOVC_ovc_behav).

Built on foragingABC's pose chain (foragingabc.dj_pipeline.pose). Planned
tables (sound stimuli, dark periods, speed, escape trials) are described in
docs/BEHAVIOUR_DJ_PLAN.md, Step 3.
"""

import datajoint as dj

from aeon.dj_pipeline import get_schema_name
from foragingabc.dj_pipeline import pose
from foragingabc.dj_pipeline.utils.sleap import pose_qc

from .heading import compute_ear_heading

schema = dj.Schema(get_schema_name("ovc_behav"))


@schema
class EarHeading(dj.Computed):
    definition = """
    -> pose.PoseQC   # run QC first, dependency for ordering/provenance
    ---
    heading_data: <xarray@dj_store>  # ear-axis heading (0 deg = up, clockwise) + confidence
    """

    def make(self, key):
        ds = (pose.SleapRawTracking & key).fetch1("pose_data")

        # same preprocessing as pose.HeadingDirection
        qc_params = (pose.PoseQC & key).fetch1("qc_params")
        ds_clean = pose_qc.quality_control_pose(ds=ds, **qc_params)

        ds_heading = compute_ear_heading(ds=ds_clean, raw_attrs=ds.attrs)
        self.insert1({**key, "heading_data": ds_heading})
