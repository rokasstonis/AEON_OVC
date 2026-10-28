"""Animal kinematics and object-relative variables in the arena cm frame.

Conventions (see geometry.py): cm frame from foragingABC's cm_convert (origin bottom-left,
y up); angles in degrees, 0 = +x, counter-clockwise, [-180, 180). Egocentric bearing =
allocentric bearing - direction, wrapped, so 0 = straight ahead and positive = to the left.

Points:
- centroid : foragingABC's weighted keypoint centroid (same as pose.PositionCM)
- head     : ear midpoint; used for all object metrics and bearings
Directions:
- head     : perpendicular to the ear axis, sign-checked against nose/head (heading.py)
- body     : mid(spine3, spine4) -> spine1
- movement : direction of the centroid velocity, only where speed >= MOVEMENT_MIN_SPEED
"""

import json

import numpy as np
import xarray as xr

from .geometry import (
    derivative,
    direction_deg,
    points_in_polygon,
    signed_distance_to_polygon,
    time_seconds,
    unwrap_deg,
    wrap180,
)
from .heading import SIGN_FALLBACK_KP, SIGN_REF_KPS, ear_confidence, ear_forward_vector

SIGMA_FRAMES = 5  # Gaussian sigma for derivatives, as foragingABC compute_velocity
SAVGOL_WINDOW = 11  # position / direction smoothing, as foragingABC position + heading
SAVGOL_POLYORDER = 2
SAVGOL_MAX_GAP = 3
MAX_INTERP_GAP = 3  # frames bridged before differentiating
MOVEMENT_MIN_SPEED = 5.0  # cm/s; movement direction is NaN below this
BODY_FRONT_KP = "spine1"
BODY_BACK_KPS = ["spine3", "spine4"]

PARAMS = {
    "sigma_frames": SIGMA_FRAMES,
    "savgol_window": SAVGOL_WINDOW,
    "savgol_polyorder": SAVGOL_POLYORDER,
    "savgol_max_gap": SAVGOL_MAX_GAP,
    "max_interp_gap": MAX_INTERP_GAP,
    "movement_min_speed_cm_s": MOVEMENT_MIN_SPEED,
    "angle_convention": "deg, 0 = +x, counter-clockwise, [-180, 180), arena cm frame (y up)",
    "egocentric_convention": "allocentric - direction, wrapped; 0 = ahead, + = left",
}


def _savgol(y):
    from foragingabc.dj_pipeline.utils.sleap.position import _savgol_nan_safe

    return _savgol_nan_safe(y, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, max_gap=SAVGOL_MAX_GAP)


def _savgol_angle(angle_deg):
    from foragingabc.dj_pipeline.utils.sleap.heading_direction import _savgol_circular

    return wrap180(_savgol_circular(angle_deg, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER,
                                    max_gap=SAVGOL_MAX_GAP))


def _nanmean_points(arrays):
    """Mean of (T, 2) arrays ignoring NaN, NaN where all are missing."""
    stack = np.stack(arrays, axis=0)
    n = (~np.isnan(stack[..., 0])).sum(axis=0)
    total = np.nansum(stack, axis=0)
    return np.where(n[:, None] > 0, total / np.maximum(n, 1)[:, None], np.nan)


def kinematics_from_cm(times, keypoints_cm, centroid_cm):
    """Core kinematics from cm keypoints (no DB, no calibration).

    times        : (T,) datetime64
    keypoints_cm : dict keypoint name -> (T, 2) cm positions (QC'd, unsmoothed)
    centroid_cm  : (T, 2) centroid in cm (already smoothed, as PositionCM)

    Returns xr.Dataset on `times`.
    """
    t_s = time_seconds(times)
    kp = keypoints_cm.get

    # head position: smoothed ear midpoint
    left, right = kp("left_ear"), kp("right_ear")
    ear_mid = (left + right) / 2
    head = np.stack([_savgol(ear_mid[:, 0]), _savgol(ear_mid[:, 1])], axis=1)

    # head direction: ear-axis perpendicular, y-up frame
    fw = ear_forward_vector(left, right, [kp(k) for k in SIGN_REF_KPS], kp(SIGN_FALLBACK_KP), y_down=False)
    head_dir_raw = np.where(fw["valid"], direction_deg(fw["f"]), np.nan)
    head_dir = _savgol_angle(head_dir_raw)
    head_conf = ear_confidence(fw["e"], fw["valid"], fw["has_ref"], fw["used_fallback"])

    # body direction: lower back -> upper back
    back = _nanmean_points([kp(k) for k in BODY_BACK_KPS if kp(k) is not None])
    body_vec = kp(BODY_FRONT_KP) - back
    body_dir = _savgol_angle(direction_deg(body_vec))

    # centroid velocity, speed, movement direction
    vx = derivative(centroid_cm[:, 0], t_s, SIGMA_FRAMES, MAX_INTERP_GAP)
    vy = derivative(centroid_cm[:, 1], t_s, SIGMA_FRAMES, MAX_INTERP_GAP)
    speed = np.hypot(vx, vy)
    movement_dir = np.where(speed >= MOVEMENT_MIN_SPEED, direction_deg(np.stack([vx, vy], axis=1)), np.nan)

    # angular head velocity from the unwrapped, smoothed head direction
    ahv = derivative(unwrap_deg(head_dir, MAX_INTERP_GAP), t_s, SIGMA_FRAMES, MAX_INTERP_GAP)

    return xr.Dataset(
        {
            "centroid_position": (("time", "space"), np.asarray(centroid_cm, dtype=float)),
            "head_position": (("time", "space"), head),
            "velocity": (("time", "space"), np.stack([vx, vy], axis=1)),
            "speed": ("time", speed),
            "head_direction": ("time", head_dir),
            "head_direction_confidence": ("time", head_conf),
            "body_direction": ("time", body_dir),
            "movement_direction": ("time", movement_dir),
            "angular_head_velocity": ("time", ahv),
        },
        coords={"time": np.asarray(times), "space": ["x", "y"]},
        attrs={
            **PARAMS,
            "units": "position cm, velocity/speed cm/s, directions deg, angular_head_velocity deg/s",
            "head_point": "ear midpoint (Savitzky-Golay smoothed)",
            "body_direction": f"mean({', '.join(BODY_BACK_KPS)}) -> {BODY_FRONT_KP}",
        },
    )


def compute_kinematics(position_px, camera, raw_attrs=None, key=None, crop=None):
    """Kinematics from QC'd pixel poses (movement format, possibly padded with neighbouring chunks).

    position_px : DataArray (time, space, keypoint, individual), QC'd, single animal
    camera      : camera name for cm_convert (e.g. 'CameraTop')
    crop        : optional (start, end) datetime64 to keep after computing (drops the padding)
    """
    from foragingabc.dj_pipeline.utils.sleap import cm_convert, position

    position_px = position_px.sortby("time")
    _, unique_idx = np.unique(position_px["time"].values, return_index=True)
    position_px = position_px.isel(time=np.sort(unique_idx))

    ds_px = xr.Dataset({"position": position_px})
    centroid_px = position.compute_position(ds=ds_px, key=key or {}, raw_attrs=raw_attrs)["position"]
    centroid_cm = cm_convert.convert_position_to_cm(centroid_px, camera=camera)
    centroid_cm = centroid_cm.isel(keypoint=0, individual=0).transpose("time", "space").values

    pos_cm = cm_convert.convert_position_to_cm(position_px, camera=camera)
    pos_cm = pos_cm.isel(individual=0).transpose("time", "keypoint", "space")
    keypoints_cm = {str(k): pos_cm.sel(keypoint=k).values for k in pos_cm["keypoint"].values}

    ds = kinematics_from_cm(position_px["time"].values, keypoints_cm, centroid_cm)
    if crop is not None:
        ds = ds.sel(time=slice(np.datetime64(crop[0]), np.datetime64(crop[1])))
    attrs = raw_attrs or {}
    ds.attrs.update({
        "camera": camera,
        "fps": attrs.get("fps"),
        "video_path": attrs.get("video_path"),
        "source_software": attrs.get("source_software"),
    })
    return ds


def compute_object_kinematics(kin, polygon_cm, centre_cm, window=None):
    """Head-relative variables for one object.

    kin        : Kinematics dataset (kinematics_from_cm / compute_kinematics)
    polygon_cm : (N, 2) object polygon in cm
    centre_cm  : (2,) object centre in cm
    window     : optional (onset, offset) datetime64 (offset None = open); applied after
                 differentiating, so the radial velocity has no artificial edges inside the chunk
    """
    head = kin["head_position"].transpose("time", "space").values
    centre = np.asarray(centre_cm, dtype=float)
    t_s = time_seconds(kin["time"].values)

    rel = head - centre  # object-centred head position
    dist_centre = np.hypot(rel[:, 0], rel[:, 1])
    dist_edge = signed_distance_to_polygon(head, polygon_cm)
    allo = direction_deg(centre - head)
    radial_velocity = derivative(dist_centre, t_s, SIGMA_FRAMES, MAX_INTERP_GAP)

    ds = xr.Dataset(
        {
            "distance_to_centre": ("time", dist_centre),
            "distance_to_edge": ("time", dist_edge),
            "inside": ("time", points_in_polygon(head, polygon_cm)),
            "object_centred_position": (("time", "space"), rel),
            "allocentric_bearing": ("time", allo),
            "egocentric_bearing_head": ("time", wrap180(allo - kin["head_direction"].values)),
            "egocentric_bearing_body": ("time", wrap180(allo - kin["body_direction"].values)),
            "egocentric_bearing_movement": ("time", wrap180(allo - kin["movement_direction"].values)),
            "radial_velocity": ("time", radial_velocity),
        },
        coords={"time": kin["time"].values, "space": ["x", "y"]},
        attrs={
            **PARAMS,
            "units": "distances cm (distance_to_edge < 0 inside), bearings deg, radial_velocity cm/s",
            "radial_velocity": "d distance_to_centre / dt; < 0 approach, > 0 retreat",
            "point": "head (ear midpoint)",
            "object_centre_cm": [float(centre[0]), float(centre[1])],
        },
    )
    if window is not None:
        onset, offset = window
        times = ds["time"].values
        keep = times >= np.datetime64(onset)
        if offset is not None:
            keep &= times < np.datetime64(offset)
        ds = ds.isel(time=np.flatnonzero(keep))
    return ds


def _safe_attrs(attrs):
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        elif isinstance(v, (list, tuple, dict)):
            flat = np.asarray(v, dtype=object).ravel() if not isinstance(v, dict) else None
            if flat is None or not all(isinstance(x, (int, float, np.integer, np.floating)) for x in flat):
                v = json.dumps(v, default=str)
        out[k] = v
    return out


def netcdf_safe(ds):
    """Copy of `ds` whose attrs NetCDF can store (the <xarray> codec writes NetCDF-4):
    None dropped, bools as int, non-numeric lists/dicts as JSON strings."""
    ds = ds.copy()
    ds.attrs = _safe_attrs(ds.attrs)
    for name in ds.variables:
        ds[name].attrs = _safe_attrs(ds[name].attrs)
    return ds
