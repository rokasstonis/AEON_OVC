"""OVC behaviour schema (aeon_foragingOVC_ovc_behav).

Built on foragingABC's pose chain (foragingabc.dj_pipeline.pose):
- EarHeading       : ear-axis heading in image coordinates (kept for comparison)
- ObjectRegistration (+ .Object) : shelter/object polygons and presence intervals (registration.py)
- SoundStimulus (+ .Event)       : sound-card play events with nominal offsets (stimuli.py)
- Kinematics       : centroid/head position, speed, head/body/movement direction, angular head
                     velocity in the arena cm frame (kinematics.py)
- ObjectKinematics : head relative to each present object: distances, bearings, radial velocity
- EscapeTrial (+ .Object) : per-stimulus response stats and a trajectory window (no escape label)

Angles: degrees, 0 = +x, counter-clockwise, [-180, 180) in the cm frame (y up); egocentric =
allocentric - direction, wrapped (0 = ahead, + = left). See docs/BEHAVIOUR_DJ_PLAN.md.
"""

import json

import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr

from aeon.dj_pipeline import acquisition, get_schema_name, streams
from foragingabc.dj_pipeline import pose
from foragingabc.dj_pipeline.utils.behaviour import get_neighbor_chunk_key
from foragingabc.dj_pipeline.utils.sleap import pose_qc

from . import escape, kinematics, stimuli
from .heading import compute_ear_heading

schema = dj.Schema(get_schema_name("ovc_behav"))

CAMERA = "CameraTop"
EDGE_PADDING_S = 2  # seconds of neighbouring chunks used so derivatives have no chunk-edge artefacts


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


# ---- Shelter / object layout ----


@schema
class ObjectRegistration(dj.Manual):
    definition = """  # shelter/object registration file of one experiment (ingest with registration.ingest)
    -> acquisition.Experiment
    ---
    registration_file: varchar(255)
    removal_lead_s: float32         # objects are removed this long before the next registration
    arena_centre_x_px: float32
    arena_centre_y_px: float32
    arena_radius_px: float32
    arena_centre_x_cm: float32
    arena_centre_y_cm: float32
    arena_radius_cm: float32
    """

    class Object(dj.Part):
        definition = """  # one registered shelter/object and the interval it is present
        -> master
        object_id: int16                  # registration order (by onset)
        ---
        identity: varchar(16)             # Shelter / Object
        position_category: varchar(32)    # 1 / 2 / 3 / sleep / trace / dario's bit
        label: varchar(64)                # label as written in the registration file
        object_onset: datetime(6)         # HARP time of the registration frame
        object_offset=null: datetime(6)   # removal time; null = until the end of the experiment
        merged_into_next: bool            # continues into the next registration (same identity + category)
        polygon_px: <blob>                # (N, 2) raw CameraTop pixels
        polygon_cm: <blob>                # (N, 2) arena cm
        centre_x_cm: float32              # polygon centroid
        centre_y_cm: float32
        video_path: varchar(255)
        frame_index: int32
        """


def objects_present(experiment_name, t):
    """ObjectRegistration.Object rows present at time t."""
    t = pd.Timestamp(t).to_pydatetime()
    return (
        ObjectRegistration.Object
        & {"experiment_name": experiment_name}
        & f"object_onset <= '{t}'"
        & f"(object_offset IS NULL OR object_offset > '{t}')"
    )


# ---- Sound stimuli ----


@schema
class SoundStimulus(dj.Computed):
    definition = """  # sound-card play events of one chunk
    -> streams.SoundCardSoundEvents
    """

    class Event(dj.Part):
        definition = """
        -> master
        stim_onset: datetime(6)          # HARP time the sound card started playing
        ---
        sound_index: int16
        sound_name='': varchar(64)
        sound_file='': varchar(128)
        duration_s=null: float32         # nominal, parsed from the sound bank file name
        stim_offset=null: datetime(6)    # onset + duration (the sound card logs no stop event)
        """

    def make(self, key):
        df = (streams.SoundCardSoundEvents & key).fetch1("stream_df")
        self.insert1(key)
        if df is None or len(df) == 0:
            return

        epoch_start = (acquisition.Chunk & key).fetch1("epoch_start")
        metadata = (
            acquisition.EpochConfig.Meta & {"experiment_name": key["experiment_name"], "epoch_start": epoch_start}
        ).fetch1("metadata")
        bank = stimuli.sound_bank_by_index(metadata)

        df = df[~df.index.duplicated()]
        rows = []
        for t, sound_index in zip(df.index, df["sound_index"], strict=True):
            sound = bank.get(int(sound_index), {})
            duration = sound.get("duration_s")
            onset = pd.Timestamp(t)
            rows.append({
                **key,
                "stim_onset": onset.to_pydatetime(),
                "sound_index": int(sound_index),
                "sound_name": sound.get("name", ""),
                "sound_file": sound.get("file", ""),
                "duration_s": duration,
                "stim_offset": None if duration is None else (onset + pd.Timedelta(seconds=duration)).to_pydatetime(),
            })
        self.Event.insert(rows)


# ---- Kinematics ----


def _qc_position(key):
    """QC'd pixel keypoints (movement 'position' DataArray) and the raw attrs of one pose chunk."""
    ds = (pose.SleapRawTracking & key).fetch1("pose_data")
    qc_params = (pose.PoseQC & key).fetch1("qc_params")
    if isinstance(qc_params, str):
        qc_params = json.loads(qc_params)
    return pose_qc.quality_control_pose(ds=ds, **qc_params)["position"].sortby("time"), ds.attrs


def _padded_position(key, padding_s=EDGE_PADDING_S):
    """Own chunk's QC'd keypoints plus up to `padding_s` from the neighbouring chunks."""
    own, attrs = _qc_position(key)
    times = own["time"].values
    start, end = times[0], times[-1]
    pad = np.timedelta64(int(padding_s * 1e9), "ns")
    pieces = [own]
    for direction in ("previous", "next"):
        chunk = get_neighbor_chunk_key(key, direction)
        if chunk is None:
            continue
        neighbour_key = {**key, "chunk_start": chunk["chunk_start"]}
        if not (pose.PoseQC & neighbour_key):
            continue
        pos, _ = _qc_position(neighbour_key)
        t = pos["time"].values
        keep = (t < start) & (t >= start - pad) if direction == "previous" else (t > end) & (t <= end + pad)
        if keep.any():
            pieces.append(pos.isel(time=np.flatnonzero(keep)))
    return xr.concat(pieces, dim="time").sortby("time"), attrs, (start, end)


@schema
class Kinematics(dj.Computed):
    definition = """  # centroid/head kinematics in the arena cm frame (kinematics.py)
    -> pose.PositionCM
    ---
    kinematics_data: <xarray@dj_store>  # positions, velocity, speed, head/body/movement direction, AHV
    """

    def make(self, key):
        position, attrs, crop = _padded_position(key)
        ds = kinematics.compute_kinematics(position, camera=key["device_name"], raw_attrs=attrs, key=key, crop=crop)
        self.insert1({**key, "kinematics_data": kinematics.netcdf_safe(ds)})


@schema
class ObjectKinematics(dj.Computed):
    definition = """  # head relative to one object, for this chunk's frames while the object is present
    -> Kinematics
    -> ObjectRegistration.Object
    ---
    n_frames: int32
    object_kinematics: <xarray@dj_store>  # distances, bearings, object-centred position, radial velocity
    """

    @property
    def key_source(self):
        chunks = Kinematics.proj() * acquisition.Chunk.proj("chunk_end")
        objects = ObjectRegistration.Object.proj("object_onset", "object_offset")
        return (
            chunks * objects
            & "object_onset < chunk_end"
            & "(object_offset IS NULL OR object_offset > chunk_start)"
        ).proj()

    def make(self, key):
        kin = (Kinematics & key).fetch1("kinematics_data")
        polygon, cx, cy, onset, offset = (ObjectRegistration.Object & key).fetch1(
            "polygon_cm", "centre_x_cm", "centre_y_cm", "object_onset", "object_offset"
        )
        ds = kinematics.compute_object_kinematics(kin, polygon, (cx, cy), window=(onset, offset))
        self.insert1({**key, "n_frames": ds.sizes["time"], "object_kinematics": kinematics.netcdf_safe(ds)})


# ---- Escape trials ----


@schema
class EscapeParams(dj.Lookup):
    definition = """
    escape_paramset_id: int16
    ---
    pre_s: float32             # trajectory window before onset
    post_s: float32            # trajectory window after onset; also the window for shelter/object entry
    baseline_s: float32        # baseline speed window before onset
    speed_threshold: float32   # cm/s, for latency_to_threshold_s
    """
    contents = [(1, 5.0, 15.0, 1.0, 20.0)]


def _kinematics_window(experiment_name, start, end):
    """Kinematics of CameraTop chunks overlapping [start, end], concatenated and cropped."""
    chunks = (
        acquisition.Chunk
        & {"experiment_name": experiment_name}
        & f"chunk_start <= '{pd.Timestamp(end).to_pydatetime()}'"
        & f"chunk_end >= '{pd.Timestamp(start).to_pydatetime()}'"
    )
    rows = (Kinematics & {"device_name": CAMERA} & chunks.proj()).to_dicts(order_by="chunk_start")
    if not rows:
        return None
    pieces = [r["kinematics_data"].sel(time=slice(np.datetime64(start), np.datetime64(end))) for r in rows]
    ds = xr.concat(pieces, dim="time").sortby("time")
    _, unique_idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=np.sort(unique_idx))


def _light_before(experiment_name, t):
    t = pd.Timestamp(t)
    rows = (
        acquisition.LightEvents
        & {"experiment_name": experiment_name}
        & f"chunk_start <= '{t.to_pydatetime()}'"
        & f"chunk_start >= '{(t - stimuli.LIGHT_LOOKBACK).to_pydatetime()}'"
    ).to_arrays("stream_df")
    frames = [df for df in rows if df is not None and len(df)]
    return stimuli.last_light_value(pd.concat(frames) if frames else None, t)


def _span(seconds):
    return np.timedelta64(int(seconds * 1e9), "ns")


@schema
class EscapeTrial(dj.Computed):
    definition = """  # response to one sound stimulus: stats + trajectory window (no escape label)
    -> SoundStimulus.Event
    -> EscapeParams
    ---
    sound_index: int16
    stim_offset=null: datetime(6)
    light_value=null: float32               # last LightEvents value (channels 3/7) before onset
    is_dark=null: bool                      # light_value == 0
    in_shelter_at_onset=null: bool          # head inside a present shelter's polygon at onset
    baseline_speed=null: float32            # cm/s, mean centroid speed over baseline_s before onset
    peak_speed=null: float32                # cm/s, max centroid speed from onset to offset (or post_s)
    peak_speed_latency_s=null: float32
    latency_to_threshold_s=null: float32    # first speed >= speed_threshold within post_s
    n_frames: int32
    trajectory: <xarray@dj_store>           # -pre_s..+post_s kinematics + per-object variables
    """

    class Object(dj.Part):
        definition = """  # each shelter/object present at onset
        -> master
        -> ObjectRegistration.Object
        ---
        identity: varchar(16)
        position_category: varchar(32)
        distance_to_centre_at_onset=null: float32       # cm, head
        distance_to_edge_at_onset=null: float32         # cm, < 0 inside
        egocentric_bearing_head_at_onset=null: float32  # deg
        egocentric_bearing_body_at_onset=null: float32
        egocentric_bearing_movement_at_onset=null: float32
        min_distance_to_edge=null: float32              # cm, onset -> onset + post_s
        time_to_enter_s=null: float32                   # head first inside after onset (null if not, or inside at onset)
        """

    @property
    def key_source(self):
        # the stimulus chunk needs CameraTop kinematics; join on experiment + chunk only
        has_kinematics = (Kinematics & {"device_name": CAMERA}).proj(camera="device_name")
        return SoundStimulus.Event.proj() * EscapeParams & has_kinematics

    def make(self, key):
        exp = key["experiment_name"]
        sound_index, stim_offset = (SoundStimulus.Event & key).fetch1("sound_index", "stim_offset")
        pre_s, post_s, baseline_s, speed_threshold = (EscapeParams & key).fetch1(
            "pre_s", "post_s", "baseline_s", "speed_threshold"
        )
        onset = np.datetime64(pd.Timestamp(key["stim_onset"]), "ns")
        offset = None if stim_offset is None else np.datetime64(pd.Timestamp(stim_offset), "ns")
        light = _light_before(exp, onset)
        row = {
            **key,
            "sound_index": sound_index,
            "stim_offset": stim_offset,
            "light_value": light,
            "is_dark": None if light is None else light == 0,
        }

        kin = _kinematics_window(exp, onset - _span(pre_s), onset + _span(post_s))
        if kin is None or kin.sizes["time"] == 0:
            self.insert1({**row, "n_frames": 0, "trajectory": xr.Dataset()})
            return

        objects = objects_present(exp, key["stim_onset"]).to_dicts()
        stats, object_stats, trajectory = escape.trial_stats(
            kin, onset, offset, objects, pre_s, post_s, baseline_s, speed_threshold
        )
        self.insert1({**row, **stats, "trajectory": kinematics.netcdf_safe(trajectory)})
        if object_stats:
            self.Object.insert([{**key, **o} for o in object_stats])
