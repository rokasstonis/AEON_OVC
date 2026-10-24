"""Heading direction from the two ears.

Alternative to foragingABC's spine -> head-centre heading
(foragingabc/dj_pipeline/utils/sleap/heading_direction.py). The heading is
perpendicular to the left_ear -> right_ear axis, pointing forward from the ear
midpoint. Output uses the same variable names, angle convention (0 deg = up,
clockwise, image coordinates) and circular Savitzky-Golay smoothing as
foragingABC, so the two are interchangeable downstream.
"""

import numpy as np
import xarray as xr

from foragingabc.dj_pipeline.utils.sleap.heading_direction import _savgol_circular
from foragingabc.dj_pipeline.utils.sleap.utils import die, print_warn

EAR_KPS = ["left_ear", "right_ear"]
# Keypoints used, in order of preference, to decide which of the two
# perpendiculars is "forward" (guards against swapped left/right ear labels).
SIGN_REF_KPS = ["nose", "head"]
SIGN_FALLBACK_KP = "spine1"  # behind the ears, so ref = ear_mid - spine1


def compute_ear_heading(ds: xr.Dataset,
                        raw_attrs: dict | None = None,
                        savgol_window: int = 11,
                        savgol_polyorder: int = 2,
                        savgol_max_gap: int = 3) -> xr.Dataset:
    """
    Compute heading direction from the ear axis.

    Parameters
    ----------
    ds : QC-cleaned movement poses dataset (pose_qc.quality_control_pose output),
        'position' with dims (time, space, keypoint, individual).

    Returns
    -------
    xr.Dataset with:
        heading_direction : (time, individual) - degrees 0-360, 0 = up, clockwise
        confidence        : (time, individual) - 0-1
    """
    try:
        pos = ds["position"].transpose("time", "space", "keypoint", "individual")
        keypoints = list(ds.coords["keypoint"].values)
    except Exception as exc:
        die(f"Failed to read 'position' data from dataset: {exc}")

    missing = [kp for kp in EAR_KPS if kp not in keypoints]
    if missing:
        die(f"Ear heading needs keypoints {EAR_KPS}; missing {missing}")

    pos_np = pos.values  # (T, 2, K, I)
    T, _, _, I = pos_np.shape

    def _kp(name):
        return pos_np[:, :, keypoints.index(name), :] if name in keypoints else None  # (T, 2, I)

    left, right = _kp("left_ear"), _kp("right_ear")
    refs = [_kp(kp) for kp in SIGN_REF_KPS]
    fallback = _kp(SIGN_FALLBACK_KP)

    heading_out = np.full((T, I), np.nan)
    confidence_out = np.full((T, I), np.nan)
    flip_frac = []

    for ii in range(I):
        L, R = left[:, :, ii], right[:, :, ii]  # (T, 2)
        ear_mid = (L + R) / 2
        e = R - L

        # --- forward = perpendicular to the ear axis ---
        # Image coords (y down): facing up -> e = (+1, 0) -> f = (0, -1)
        f = np.stack([e[:, 1], -e[:, 0]], axis=1)  # (T, 2)
        valid = ~np.isnan(f).any(axis=1)

        # --- sign check: forward must point towards the snout ---
        ref = np.full((T, 2), np.nan)
        for arr in refs:
            if arr is None:
                continue
            fill = np.isnan(ref).any(axis=1)
            ref[fill] = arr[fill, :, ii] - ear_mid[fill]
        used_fallback = np.isnan(ref).any(axis=1)
        if fallback is not None:
            ref[used_fallback] = ear_mid[used_fallback] - fallback[used_fallback, :, ii]
        has_ref = ~np.isnan(ref).any(axis=1)

        flip = valid & has_ref & ((f * ref).sum(axis=1) < 0)
        f[flip] *= -1
        n_checked = int((valid & has_ref).sum())
        flip_frac.append(float(flip.sum() / n_checked) if n_checked else float("nan"))

        # --- angle: 0 = up/north, clockwise, image coords (y downward) ---
        angle = (-np.degrees(np.arctan2(-f[:, 1], f[:, 0])) + 90) % 360
        angle[~valid] = np.nan

        # --- confidence: apparent ear separation relative to typical ---
        # Ears close together = head tilted/turned or an ear misplaced.
        ear_sep = np.sqrt((e ** 2).sum(axis=1))
        median_sep = np.nanmedian(ear_sep[valid]) if valid.any() else np.nan
        conf = np.clip(ear_sep / median_sep, 0, 1) if median_sep > 0 else np.zeros(T)
        conf = np.where(valid & used_fallback, conf / 2, conf)  # sign from spine1 only
        conf = np.where(valid & ~has_ref, 0.0, conf)  # sign could not be checked
        conf[~valid] = 0.0

        n_invalid = int((~valid).sum())
        if n_invalid:
            print_warn(f"Individual {ii}: {n_invalid} frame(s) without both ears — NaN heading.")

        # --- smooth angle (circular), as in foragingABC ---
        heading_out[:, ii] = _savgol_circular(angle, window=savgol_window,
                                              polyorder=savgol_polyorder,
                                              max_gap=savgol_max_gap) % 360
        confidence_out[:, ii] = conf

    attrs = raw_attrs or {}
    return xr.Dataset(
        {
            "heading_direction": (("time", "individual"), heading_out),
            "confidence": (("time", "individual"), confidence_out),
        },
        coords={
            "time": ds.coords["time"].values,
            "individual": ds.coords["individual"].values,
        },
        attrs={
            "source_software": attrs.get("source_software"),
            "fps": attrs.get("fps"),
            "time_unit": attrs.get("time_unit"),
            "video_path": attrs.get("video_path"),
            "log": attrs.get("log", []),
            "method": "ears",
            "ear_kps": EAR_KPS,
            "sign_ref_kps": SIGN_REF_KPS + [SIGN_FALLBACK_KP],
            "ear_sign_flip_frac": flip_frac,
            "savgol_window": savgol_window,
            "savgol_polyorder": savgol_polyorder,
            "savgol_max_gap": savgol_max_gap,
        },
    )
