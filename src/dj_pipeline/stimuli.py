"""Sound-bank and light helpers for SoundStimulus / EscapeTrial (no table definitions)."""

import json
import re

import numpy as np
import pandas as pd

LIGHT_CHANNELS = (3, 7)  # Environment LightEvents channels driving the arena light; 0 = dark
LIGHT_LOOKBACK = pd.Timedelta(hours=3)  # light is logged ~every 60 s, but not at all while dark


def find_sound_bank(metadata):
    """Find the rig's soundBank list anywhere in an epoch metadata dict (or JSON string)."""
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    stack = [metadata]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() == "soundbank" and isinstance(v, list):
                    return v
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return []


def sound_duration_s(file_path, sound_name=""):
    """Nominal duration from the sound file name: the last `_<N>s` before `_` or `.`
    ('..._3s_sweep_x3_192kHz_24bit_9s.wav' -> 9), else `<A>x<B>s` / `_<N>s` at the end of the
    sound name ('17kHz_to_20kHz_3x3s' -> 9). None if neither matches."""
    found = re.findall(r"_(\d+(?:\.\d+)?)s(?=[_.])", file_path or "")
    if found:
        return float(found[-1])
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)s$", sound_name or "")
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"_(\d+(?:\.\d+)?)s$", sound_name or "")
    return float(m.group(1)) if m else None


def sound_bank_by_index(metadata):
    """{sound_index: {"name", "file", "duration_s"}} from epoch metadata."""
    bank = {}
    for s in find_sound_bank(metadata):
        params = s.get("soundParameters", {})
        if "soundIndex" not in params:
            continue
        file_path = s.get("filePath", "")
        bank[int(params["soundIndex"])] = {
            "name": s.get("soundName", ""),
            "file": file_path,
            "duration_s": sound_duration_s(file_path, s.get("soundName", "")),
        }
    return bank


def last_light_value(light_df, t):
    """Last LightEvents value on LIGHT_CHANNELS at or before `t` (None if none logged).

    light_df: concatenated Environment LightEvents stream (time index, columns channel, value).
    """
    if light_df is None or len(light_df) == 0:
        return None
    df = light_df[light_df["channel"].isin(LIGHT_CHANNELS)].sort_index()
    df = df[df.index <= pd.Timestamp(t)]
    return None if df.empty else float(df["value"].iloc[-1])


def first_time(times, mask):
    """First time where mask is True, or None."""
    idx = np.flatnonzero(np.asarray(mask))
    return None if idx.size == 0 else times[idx[0]]
