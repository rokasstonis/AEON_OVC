"""Angle, polygon and derivative helpers for kinematics (numpy only, no DB).

Angles are degrees in the arena cm frame (y up): 0 = +x, counter-clockwise,
wrapped to [-180, 180).
"""

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d


def wrap180(angle_deg):
    """Wrap degrees to [-180, 180). NaN stays NaN."""
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def direction_deg(vec):
    """Direction of (..., 2) vectors in degrees, 0 = +x, counter-clockwise, [-180, 180)."""
    vec = np.asarray(vec, dtype=float)
    return wrap180(np.degrees(np.arctan2(vec[..., 1], vec[..., 0])))


def polygon_centroid(poly):
    """Area centroid of a simple polygon (N, 2); falls back to the vertex mean if degenerate."""
    poly = np.asarray(poly, dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * area)


def signed_distance_to_polygon(points, poly):
    """Distance from (T, 2) points to the polygon's boundary; negative inside. NaN in -> NaN out."""
    points = np.asarray(points, dtype=float)
    poly = np.asarray(poly, dtype=float)
    out = np.full(len(points), np.nan)
    ok = ~np.isnan(points).any(axis=1)
    if not ok.any():
        return out
    p = points[ok][:, None, :]  # (T, 1, 2)
    a = poly[None, :, :]  # (1, E, 2)
    b = np.roll(poly, -1, axis=0)[None, :, :]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
    dist = np.sqrt((((a + t[..., None] * ab) - p) ** 2).sum(-1)).min(axis=1)
    inside = Path(poly).contains_points(points[ok])
    out[ok] = np.where(inside, -dist, dist)
    return out


def points_in_polygon(points, poly):
    """Boolean (T,) - point inside the polygon; False for NaN points."""
    points = np.asarray(points, dtype=float)
    out = np.zeros(len(points), dtype=bool)
    ok = ~np.isnan(points).any(axis=1)
    if ok.any():
        out[ok] = Path(np.asarray(poly, dtype=float)).contains_points(points[ok])
    return out


def fill_short_gaps(y, max_gap):
    """Linearly interpolate NaN runs of at most `max_gap` samples (1-D); longer runs stay NaN."""
    y = np.asarray(y, dtype=float).copy()
    nan = np.isnan(y)
    if not nan.any() or nan.all():
        return y
    idx = np.arange(len(y))
    filled = np.interp(idx, idx[~nan], y[~nan])
    # length of each NaN run, assigned to its samples
    edges = np.diff(np.concatenate([[0], nan.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for s, e in zip(starts, ends, strict=True):
        if e - s <= max_gap and s > 0 and e < len(y):
            y[s:e] = filled[s:e]
    return y


def smooth_nan(y, sigma):
    """Gaussian smoothing (sigma in samples) that ignores NaN: normalised convolution; NaN stays NaN."""
    y = np.asarray(y, dtype=float)
    if sigma <= 0:
        return y.copy()
    nan = np.isnan(y)
    num = gaussian_filter1d(np.where(nan, 0.0, y), sigma, mode="nearest")
    den = gaussian_filter1d((~nan).astype(float), sigma, mode="nearest")
    out = num / np.where(den > 1e-6, den, np.nan)
    out[nan] = np.nan
    return out


def time_seconds(times):
    """datetime64 array -> float seconds from its first sample."""
    times = np.asarray(times)
    return (times - times[0]) / np.timedelta64(1, "s")


def derivative(y, t_s, sigma=0, max_gap=3):
    """d y / dt for a 1-D series with NaN gaps: short gaps interpolated, np.gradient on
    `t_s` seconds, then NaN-aware Gaussian smoothing (sigma in samples). Long gaps stay NaN."""
    y = fill_short_gaps(y, max_gap)
    if len(y) < 2:
        return np.full(len(y), np.nan)
    return smooth_nan(np.gradient(y, t_s), sigma)


def unwrap_deg(angle_deg, max_gap=3):
    """Unwrap degrees across NaN gaps: short gaps interpolated on the circle; each valid run is
    unwrapped and continues from the previous run's last value."""
    a = np.asarray(angle_deg, dtype=float)
    out = np.full(len(a), np.nan)
    rad = np.deg2rad(a)
    s = fill_short_gaps(np.sin(rad), max_gap)
    c = fill_short_gaps(np.cos(rad), max_gap)
    ang = np.arctan2(s, c)
    ok = ~np.isnan(ang)
    if ok.any():
        out[ok] = np.degrees(np.unwrap(ang[ok]))
    return out
