"""Per-stimulus response statistics for ovc_behav.EscapeTrial (no DB access, no escape label)."""

import numpy as np
import xarray as xr

from .kinematics import compute_object_kinematics
from .stimuli import first_time

ONSET_VARS = (
    "distance_to_centre",
    "distance_to_edge",
    "egocentric_bearing_head",
    "egocentric_bearing_body",
    "egocentric_bearing_movement",
)


def _span(seconds):
    return np.timedelta64(int(seconds * 1e9), "ns")


def _seconds(t, onset):
    return None if t is None else float((t - onset) / np.timedelta64(1, "s"))


def _value(x):
    x = float(x)
    return None if np.isnan(x) else x


def trial_stats(kin, onset, offset, objects, pre_s, post_s, baseline_s, speed_threshold):
    """Response statistics for one stimulus.

    kin     : Kinematics dataset covering at least onset - pre_s .. onset + post_s
    onset   : datetime64 stimulus onset; offset: datetime64 or None (then onset + post_s)
    objects : dicts with object_id, identity, position_category, polygon_cm, centre_x_cm, centre_y_cm
              (the objects present at onset)

    Returns (stats, object_stats, trajectory):
      stats        : dict of EscapeTrial secondary attributes (None where undefined)
      object_stats : one dict per object (EscapeTrial.Object secondary attributes + object_id)
      trajectory   : kin cropped to the window, per-object variables prefixed "object_" on an
                     object_id dim, and a time_from_onset_s coordinate
    """
    onset = np.datetime64(onset, "ns")
    response_end = onset + _span(post_s) if offset is None else np.datetime64(offset, "ns")
    kin = kin.sel(time=slice(onset - _span(pre_s), onset + _span(post_s)))
    times, speed = kin["time"].values, kin["speed"].values

    baseline = (times >= onset - _span(baseline_s)) & (times < onset)
    response = (times >= onset) & (times <= response_end)
    post = (times >= onset) & (times <= onset + _span(post_s))
    after = np.flatnonzero(times >= onset)
    i0 = after[0] if after.size else None

    stats = {
        "n_frames": int(len(times)),
        "baseline_speed": None,
        "peak_speed": None,
        "peak_speed_latency_s": None,
        "in_shelter_at_onset": None,
        "latency_to_threshold_s": _seconds(first_time(times, post & (speed >= speed_threshold)), onset),
    }
    if baseline.any() and not np.isnan(speed[baseline]).all():
        stats["baseline_speed"] = float(np.nanmean(speed[baseline]))
    if response.any() and not np.isnan(speed[response]).all():
        i_peak = np.flatnonzero(response)[np.nanargmax(speed[response])]
        stats["peak_speed"] = float(speed[i_peak])
        stats["peak_speed_latency_s"] = _seconds(times[i_peak], onset)

    object_stats, object_ds = [], []
    for obj in objects:
        ods = compute_object_kinematics(kin, obj["polygon_cm"], (obj["centre_x_cm"], obj["centre_y_cm"]))
        object_ds.append(ods.expand_dims(object_id=[obj["object_id"]]))
        o = {
            "object_id": obj["object_id"],
            "identity": obj["identity"],
            "position_category": obj["position_category"],
            "min_distance_to_edge": None,
            "time_to_enter_s": None,
        }
        for var in ONSET_VARS:
            o[f"{var}_at_onset"] = None if i0 is None else _value(ods[var].values[i0])
        edge_at_onset = o["distance_to_edge_at_onset"]
        if edge_at_onset is not None and edge_at_onset >= 0:  # outside at onset
            o["time_to_enter_s"] = _seconds(first_time(times, post & ods["inside"].values), onset)
        edge = ods["distance_to_edge"].values[post]
        if edge.size and not np.isnan(edge).all():
            o["min_distance_to_edge"] = float(np.nanmin(edge))
        if obj["identity"] == "Shelter" and edge_at_onset is not None:
            stats["in_shelter_at_onset"] = bool(stats["in_shelter_at_onset"] or edge_at_onset < 0)
        object_stats.append(o)

    trajectory = kin
    if object_ds:
        objs = xr.concat(object_ds, dim="object_id", combine_attrs="drop_conflicts")
        objs = objs.rename({v: f"object_{v}" for v in objs.data_vars})
        trajectory = xr.merge([kin, objs], combine_attrs="override")
    trajectory = trajectory.assign_coords(time_from_onset_s=("time", (times - onset) / np.timedelta64(1, "s")))
    return stats, object_stats, trajectory
