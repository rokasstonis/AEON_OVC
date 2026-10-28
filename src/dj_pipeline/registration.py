"""Shelter/object registration files -> ovc_behav.ObjectRegistration.

A registration file (``<REGISTRATION_ROOT>/<experiment>/combined_registration.json``) lists
objects registered on raw CameraTop frames: identity (Shelter/Object), label
("<Identity> - position <X>"), pixel polygon and HARP ``timestamp_s``.

Timeline rules:
- an object is present from its registration time (onset);
- it is removed ``removal_lead_s`` (5 min) before the next, later registration time;
- objects registered at the same time are all present together;
- if the next registration time includes an object with the same identity and position
  category, the object continues until that time with no gap (merged_into_next);
- the last registration time's objects stay until the end of the experiment (offset NULL);
- labels whose category is in IGNORED_CATEGORIES ("not object") are dropped entirely.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRATION_ROOT = Path("/ceph/branco/Rokas/Ephys/Results/registration_tables")
REGISTRATION_FILE = "combined_registration.json"
CAMERA = "CameraTop"
REMOVAL_LEAD_S = 300
IGNORED_CATEGORIES = {"not object"}
HARP_EPOCH = np.datetime64("1904-01-01T00:00:00", "us")


def harp_seconds_to_datetime(seconds):
    """HARP seconds (since 1904-01-01) -> datetime64[us]."""
    return HARP_EPOCH + np.timedelta64(int(round(float(seconds) * 1e6)), "us")


def parse_position_category(identity, label):
    """'Shelter - position sleep' -> 'sleep', "Object - Object - Dario's bit" -> "dario's bit"."""
    rest = label.strip()
    prefix = f"{identity} - ".lower()
    while rest.lower().startswith(prefix):
        rest = rest[len(prefix):].strip()
    rest = rest.lower()
    if rest.startswith("position "):
        rest = rest[len("position "):]
    return rest.strip()


def load_registration(path, removal_lead_s=REMOVAL_LEAD_S):
    """Parse a registration file into (arena, objects) with onset/offset applied.

    arena   : dict cx, cy, radius (pixels)
    objects : list of dicts sorted by onset, with object_id, identity, position_category, label,
              onset, offset (None = until the end), merged_into_next, polygon_px (N, 2),
              video_path, frame_index
    """
    with open(path) as f:
        reg = json.load(f)

    objects = []
    for o in reg["objects"]:
        category = parse_position_category(o["identity"], o["label"])
        if category in IGNORED_CATEGORIES:
            continue
        objects.append({
            "identity": o["identity"],
            "position_category": category,
            "label": o["label"],
            "onset": harp_seconds_to_datetime(o["timestamp_s"]),
            "polygon_px": np.asarray(o["polygon"], dtype=float),
            "video_path": o.get("video_path", ""),
            "frame_index": int(o["frame_index"]),
        })
    objects.sort(key=lambda o: o["onset"])

    onsets = sorted({o["onset"] for o in objects})
    lead = np.timedelta64(int(removal_lead_s * 1e6), "us")
    for object_id, o in enumerate(objects):
        o["object_id"] = object_id
        later = [t for t in onsets if t > o["onset"]]
        if not later:
            o["offset"], o["merged_into_next"] = None, False
            continue
        t_next = later[0]
        same = any(
            n["onset"] == t_next
            and n["identity"] == o["identity"]
            and n["position_category"] == o["position_category"]
            for n in objects
        )
        o["merged_into_next"] = same
        o["offset"] = t_next if same else max(t_next - lead, o["onset"])
    return reg["arena_boundary"], objects


def pixels_to_cm(points_px, camera=CAMERA):
    """(N, 2) raw CameraTop pixels -> (N, 2) world cm (undistort + homography, y up)."""
    from foragingabc.dj_pipeline.utils.sleap import cm_convert

    points_px = np.asarray(points_px, dtype=float)
    x_cm, y_cm = cm_convert.map_pixel_to_cm(camera, points_px[:, 0], points_px[:, 1])
    return np.stack([x_cm, y_cm], axis=1)


def arena_circle_cm(arena, n=72):
    """Arena circle (pixel centre + radius) mapped to cm: (centre_x, centre_y, radius)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = pixels_to_cm(np.stack([arena["cx"] + arena["radius"] * np.cos(theta),
                                  arena["cy"] + arena["radius"] * np.sin(theta)], axis=1))
    centre = ring.mean(axis=0)
    return float(centre[0]), float(centre[1]), float(np.linalg.norm(ring - centre, axis=1).mean())


def _py_datetime(t):
    return None if t is None else pd.Timestamp(t).to_pydatetime()


def ingest(experiment_name, path=None, removal_lead_s=REMOVAL_LEAD_S, replace=False):
    """Insert one experiment's registration file into ObjectRegistration (+ .Object).

    path defaults to REGISTRATION_ROOT/<experiment_name>/combined_registration.json.
    An existing entry is kept unless replace=True, which deletes it first (cascades to
    ObjectKinematics and EscapeTrial).
    """
    from .geometry import polygon_centroid
    from .ovc_behav import ObjectRegistration

    path = Path(path) if path else REGISTRATION_ROOT / experiment_name / REGISTRATION_FILE
    key = {"experiment_name": experiment_name}
    if ObjectRegistration & key:
        if not replace:
            print(f"ObjectRegistration already has {experiment_name}; pass replace=True to re-ingest.")
            return
        (ObjectRegistration & key).delete()

    arena, objects = load_registration(path, removal_lead_s)
    arena_x, arena_y, arena_r = arena_circle_cm(arena)
    rows = []
    for o in objects:
        polygon_cm = pixels_to_cm(o["polygon_px"])
        centre = polygon_centroid(polygon_cm)
        rows.append({
            **key,
            "object_id": o["object_id"],
            "identity": o["identity"],
            "position_category": o["position_category"],
            "label": o["label"],
            "object_onset": _py_datetime(o["onset"]),
            "object_offset": _py_datetime(o["offset"]),
            "merged_into_next": o["merged_into_next"],
            "polygon_px": o["polygon_px"],
            "polygon_cm": polygon_cm,
            "centre_x_cm": float(centre[0]),
            "centre_y_cm": float(centre[1]),
            "video_path": o["video_path"],
            "frame_index": o["frame_index"],
        })

    conn = ObjectRegistration.connection
    with conn.transaction:
        ObjectRegistration.insert1({
            **key,
            "registration_file": str(path),
            "removal_lead_s": removal_lead_s,
            "arena_centre_x_px": float(arena["cx"]),
            "arena_centre_y_px": float(arena["cy"]),
            "arena_radius_px": float(arena["radius"]),
            "arena_centre_x_cm": arena_x,
            "arena_centre_y_cm": arena_y,
            "arena_radius_cm": arena_r,
        })
        ObjectRegistration.Object.insert(rows)
    print(f"Ingested {len(rows)} objects for {experiment_name} from {path}")
