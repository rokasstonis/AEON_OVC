"""Fetch OVC behaviour across chunks (HARP clock, same as spike times).

    from src.dj_pipeline import behaviour_queries as bq

    exp = "ProjectAeonOVC-2026-09-14_001"
    kin = bq.fetch_kinematics(exp, "2026-09-15 10:00", "2026-09-15 10:30")    # xr.Dataset
    objs = bq.fetch_object_kinematics(exp, "2026-09-15 10:00", "2026-09-15 10:30", identity="Shelter")
    # -> {object_id: xr.Dataset}; two objects present at once give two entries
    stims = bq.fetch_sound_events(exp)                                          # DataFrame
    trials = bq.fetch_escape_trials(exp)                                        # one row per stimulus
    traj = bq.fetch_trial_trajectory(exp, trials.iloc[0]["stim_onset"])         # xr.Dataset

Variables and conventions are documented in kinematics.py and ovc_behav.py.
"""

import numpy as np
import pandas as pd
import xarray as xr

from aeon.dj_pipeline import acquisition

from .ovc_behav import CAMERA, EscapeTrial, Kinematics, ObjectKinematics, ObjectRegistration, SoundStimulus


def _chunks(experiment_name, start=None, end=None):
    q = acquisition.Chunk & {"experiment_name": experiment_name}
    if end is not None:
        q &= f"chunk_start <= '{pd.Timestamp(end).to_pydatetime()}'"
    if start is not None:
        q &= f"chunk_end >= '{pd.Timestamp(start).to_pydatetime()}'"
    return q.proj()


def _concat(datasets, start, end):
    datasets = [ds.sel(time=slice(start and np.datetime64(pd.Timestamp(start)),
                                  end and np.datetime64(pd.Timestamp(end)))) for ds in datasets]
    datasets = [ds for ds in datasets if ds.sizes.get("time", 0)]
    if not datasets:
        return None
    ds = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts").sortby("time")
    _, unique_idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=np.sort(unique_idx))


def fetch_kinematics(experiment_name, start=None, end=None, camera=CAMERA):
    """Kinematics between start and end (whole experiment if both None), one Dataset."""
    rows = (Kinematics & {"device_name": camera} & _chunks(experiment_name, start, end)).to_dicts(
        order_by="chunk_start"
    )
    return _concat([r["kinematics_data"] for r in rows], start, end)


def fetch_object_kinematics(experiment_name, start=None, end=None, identity=None, object_id=None, camera=CAMERA):
    """{object_id: Dataset} of head-relative variables between start and end."""
    q = ObjectKinematics & {"device_name": camera} & _chunks(experiment_name, start, end)
    if identity is not None:
        q &= (ObjectRegistration.Object & {"identity": identity}).proj()
    if object_id is not None:
        q &= {"object_id": object_id}
    by_object = {}
    for row in q.to_dicts(order_by="chunk_start"):
        by_object.setdefault(row["object_id"], []).append(row["object_kinematics"])
    out = {oid: _concat(parts, start, end) for oid, parts in by_object.items()}
    return {oid: ds for oid, ds in out.items() if ds is not None}


def fetch_objects(experiment_name):
    """Registered shelters/objects with their presence intervals (DataFrame)."""
    return pd.DataFrame(
        (ObjectRegistration.Object & {"experiment_name": experiment_name}).to_dicts(order_by="object_id")
    )


def fetch_sound_events(experiment_name):
    """All sound stimuli: onset, offset, index, name, duration (DataFrame)."""
    rows = (SoundStimulus.Event & {"experiment_name": experiment_name}).to_dicts(order_by="stim_onset")
    cols = ["stim_onset", "stim_offset", "sound_index", "sound_name", "duration_s", "sound_file", "chunk_start"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def fetch_escape_trials(experiment_name, escape_paramset_id=1):
    """One row per stimulus with response stats; per-object stats widened as
    '<identity>_<stat>' (shelter_time_to_enter_s, object_distance_to_edge_at_onset, ...).
    If two objects of the same identity were present, the one nearest at onset is used;
    see fetch_escape_trial_objects for all of them."""
    r = {"experiment_name": experiment_name, "escape_paramset_id": escape_paramset_id}
    stats = [a for a in EscapeTrial.heading.secondary_attributes if a != "trajectory"]  # skip the file fetch
    trials = pd.DataFrame((EscapeTrial & r).proj(*stats).to_dicts(order_by="stim_onset"))
    if trials.empty:
        return trials
    objs = fetch_escape_trial_objects(experiment_name, escape_paramset_id)
    if objs.empty:
        return trials
    objs = objs.sort_values("distance_to_edge_at_onset").drop_duplicates(["stim_onset", "identity"])
    stats = [c for c in objs.columns if c.endswith("_at_onset") or c in
             ("min_distance_to_edge", "time_to_enter_s", "object_id", "position_category")]
    wide = objs.pivot(index="stim_onset", columns="identity", values=stats)
    wide.columns = [f"{identity.lower()}_{stat}" for stat, identity in wide.columns]
    return trials.merge(wide.reset_index(), on="stim_onset", how="left")


def fetch_escape_trial_objects(experiment_name, escape_paramset_id=1):
    """Long table: one row per (stimulus, object present at onset)."""
    r = {"experiment_name": experiment_name, "escape_paramset_id": escape_paramset_id}
    return pd.DataFrame((EscapeTrial.Object & r).to_dicts(order_by="stim_onset"))


def fetch_trial_trajectory(experiment_name, stim_onset, escape_paramset_id=1):
    """Trajectory window of one trial (Dataset with time_from_onset_s coordinate)."""
    key = {"experiment_name": experiment_name, "escape_paramset_id": escape_paramset_id,
           "stim_onset": pd.Timestamp(stim_onset).to_pydatetime()}
    return (EscapeTrial & key).fetch1("trajectory")
