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


# ---- Light / dark periods ----
# Environment LightEvents logs channels 3/7 every ~60 s while the light is on. Dark episodes are a
# single 0 sample at the switch, then nothing until the light is back, so a dark period ends at
# the next non-zero sample, up to LOG_INTERVAL_S after the light actually returned.

LOG_INTERVAL_S = 60.0
MAX_LIGHT_GAP_S = 300.0  # longer silences while the light is on are logging gaps -> "unknown"


def light_periods(light_df, end_time):
    """Split the experiment into light / dark / unknown periods.

    light_df : LightEvents stream (time index, columns channel, value), any number of chunks
    end_time : end of the last ingested chunk; the final period is closed there

    Returns a list of dicts: period_start, period_end, condition ("light" | "dark" | "unknown").
    """
    if light_df is None or len(light_df) == 0:
        return []
    df = light_df[light_df["channel"].isin(LIGHT_CHANNELS)].sort_index()
    times, values = df.index.to_numpy(), df["value"].to_numpy(dtype=float)
    log_interval = pd.Timedelta(seconds=LOG_INTERVAL_S)
    max_gap = pd.Timedelta(seconds=MAX_LIGHT_GAP_S)
    periods = []

    def close(start, end, condition):
        if end > start:
            if periods and periods[-1]["condition"] == condition and periods[-1]["period_end"] == start:
                periods[-1]["period_end"] = end
            else:
                periods.append({"period_start": start, "period_end": end, "condition": condition})

    state, start, last = None, None, None
    for t, v in zip(pd.to_datetime(times), values, strict=True):
        new_state = "dark" if v == 0 else "light"
        if state == "light" and t - last > max_gap:  # silence while light: logging gap
            close(start, last + log_interval, "light")
            close(last + log_interval, t, "unknown")
            state, start = None, None
        if new_state != state:
            if state is not None:
                close(start, t, state)
            state, start = new_state, t
        last = t
    end_time = pd.Timestamp(end_time)
    if state == "light" and end_time - last > max_gap:
        close(start, last + log_interval, "light")
        close(last + log_interval, end_time, "unknown")
    else:
        close(start, max(end_time, last), state)
    return periods


def overlap(periods, start, end):
    """Parts of `periods` inside [start, end) (end None = open), as dicts with overlap_start,
    overlap_end, duration_s plus the period's own fields."""
    start = pd.Timestamp(start)
    end = None if end is None else pd.Timestamp(end)
    out = []
    for p in periods:
        s = max(pd.Timestamp(p["period_start"]), start)
        e = pd.Timestamp(p["period_end"]) if end is None else min(pd.Timestamp(p["period_end"]), end)
        if e > s:
            out.append({**p, "overlap_start": s, "overlap_end": e, "duration_s": (e - s).total_seconds()})
    return out


def condition_at(times, periods):
    """Light condition ('light' / 'dark' / 'unknown', '' outside all periods) for each time."""
    times = np.asarray(times, dtype="datetime64[ns]")
    out = np.full(len(times), "", dtype=object)
    for p in periods:
        s = np.datetime64(pd.Timestamp(p["period_start"]), "ns")
        e = np.datetime64(pd.Timestamp(p["period_end"]), "ns")
        out[(times >= s) & (times < e)] = p["condition"]
    return out
