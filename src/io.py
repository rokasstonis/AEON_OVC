from __future__ import annotations
# import csv
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import spikeinterface.core as si
from probeinterface import get_probe, read_probeinterface
from spikeinterface.preprocessing import unsigned_to_signed



def load_onix_recording(
    raw_dir: str | Path,
    probe_name: str = "Neuropixels2.0-1shank",
    manufacturer: str = "imec",
    sampling_frequency: float = 30000.0,
    n_channels: int = 384,
    dtype: str = "uint16",
    gain_to_uV: float = None,
    offset_to_uV: float = 0.0,
    probeinterface_json: Optional[str | Path] = None,
    probe_prefix: str = "NeuropixelsV2_ProbeA",
) -> si.BaseRecording:
    """
    Load one or more ONIX NP2.0 binary chunks from raw_dir as a SpikeInterface
    BaseRecording. Chunks are sorted by their numeric suffix and concatenated.

    Parameters
    ----------
    raw_dir
        Folder containing the .bin files and companion timestamp CSVs.
    probe_name
        ProbeInterface identifier, e.g. "Neuropixels2.0-1shank".
    manufacturer
        ProbeInterface manufacturer string (default "imec").
    sampling_frequency
        Hardware sampling rate in Hz (30 000 for NP2.0 via ONIX).
    n_channels
        Number of recorded channels (384 for NP2.0).
    dtype
        NumPy dtype string matching how bonsai-onix1 saved the file ("uint16").
    gain_to_uV
        Scalar to convert ADC counts → µV (≈2.34375 for NP2.0).
    offset_to_uV
        DC offset in µV after gain scaling (usually 0 for NP2.0).
    probeinterface_json
        Optional path to a ProbeInterface JSON exported from Bonsai.
        If provided, this is used instead of the default library geometry.
    probe_prefix
        Stem of the .bin filenames before the chunk suffix, e.g.
        "NeuropixelsV2_ProbeA_AmplifierData".

    Returns
    -------
    si.BaseRecording
        Recording with probe attached and gain set.
    """
    raw_dir = Path(raw_dir)

    # -- Discover chunk files --------------------------------------------------
    bin_files = sorted(
        raw_dir.glob(f"{probe_prefix}_AmplifierData_*.bin"),
        key=lambda p: _chunk_index(p),
    )
    if not bin_files:
        # Fallback: scan for any .bin matching the prefix
        bin_files = sorted(
            raw_dir.glob(f"{probe_prefix}*.bin"),
            key=lambda p: _chunk_index(p),
        )
    if not bin_files:
        raise FileNotFoundError(
            f"No .bin files matching '{probe_prefix}*' found in {raw_dir}.\n"
            f"Files present: {[f.name for f in raw_dir.glob('*.bin')]}"
        )

    print(f"Found {len(bin_files)} chunk(s):")
    for f in bin_files:
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name}  ({size_gb:.2f} GB)")

    if n_channels is None:
        raise ValueError(
            "n_channels is None. Check config acquisition.n_channels before loading binary data."
        )

    # -- Build per-chunk RecordingExtractors -----------------------------------
    # SpikeInterface's read_binary handles C-order (time_axis=0 = rows are samples)
    recordings = []
    for bf in bin_files:
        rec = si.read_binary(
            file_paths=str(bf),
            sampling_frequency=sampling_frequency,
            num_channels=n_channels,
            dtype=dtype,
            time_axis=0,          # rows = samples (ONIX layout)
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
            is_filtered=False,
        )
        recordings.append(rec)

    # -- Concatenate chunks if more than one -----------------------------------
    if len(recordings) == 1:
        recording = recordings[0]
    else:
        recording = si.concatenate_recordings(recordings)
        print(f"Concatenated {len(recordings)} chunks → "
              f"{recording.get_total_duration():.1f} s total")

    # -- Attach probe geometry -------------------------------------------------
    if probeinterface_json is not None:
        probegroup = read_probeinterface(probeinterface_json)
        probe = probegroup.probes[0]
        print(f"Loaded probe geometry from {probeinterface_json}")
    else:
        try:
            probe = get_probe(manufacturer, probe_name)
            print(f"Loaded default probe geometry: {manufacturer}/{probe_name}")
        except Exception as exc:
            probe = _build_np2_fallback_probe(probe_name, n_channels)
            warnings.warn(
                "Could not download/load probe geometry from ProbeInterface "
                f"({type(exc).__name__}: {exc}). "
                "Using built-in NP2 fallback geometry instead.",
                RuntimeWarning,
            )

    probe.set_device_channel_indices(np.arange(n_channels))
    recording = recording.set_probe(probe, in_place=False)
    recording = unsigned_to_signed(recording)

    # Summary
    dur = recording.get_total_duration()
    print(f"\nRecording loaded:")
    print(f"  Duration   : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Channels   : {recording.get_num_channels()}")
    print(f"  Sampling   : {recording.get_sampling_frequency()} Hz")
    print(f"  dtype      : {recording.get_dtype()}")
    print(f"  Probe      : {recording.get_probe()}")

    return recording



# ------------------------------------------------------------
# HarpSync loading and alignment
# ------------------------------------------------------------

# Actual ONIX PCIe clock rate measured from HarpSync data.
# The nominal spec is 250 MHz but the real rate is ~249.42 MHz.
# Using the wrong value accumulates ~20 s of error over 24 hrs.
# This constant is used as a fallback; prefer to estimate from data.
ONIX_CLOCK_HZ_NOMINAL = 249_423_527.0


def load_harpsync(
    harpsync_path: str | Path,
) -> "pd.DataFrame":
    """
    Load the ONIX HarpSync CSV file.

    The HarpSync stream is saved by bonsai-onix1 as a one-row-per-second
    table with four columns:

        Seconds         — integer Unix time (Harp clock, seconds since epoch)
        Value.Clock     — ONIX PCIe clock count at that second boundary
        Value.HubClock  — Hub chip clock count (10× lower jitter — prefer this)
        Value.HarpTime  — Harp protocol timestamp (= Seconds - 1)

    Use Value.HubClock for alignment — its jitter is ~0.06 µs vs ~0.55 µs
    for Value.Clock. The two clocks run at the same rate but differ by a
    small, variable offset.

    Parameters
    ----------
    harpsync_path
        Path to the HarpSync CSV file, e.g.
        "NeuropixelsV2_HarpSync_2026-04-02T090000Z.csv"

    Returns
    -------
    pd.DataFrame with columns [Seconds, Value.Clock, Value.HubClock,
                                Value.HarpTime] and a derived column
                                hub_clock_hz (estimated from successive diffs).
    """
    import pandas as pd

    harpsync_path = Path(harpsync_path)
    if not harpsync_path.exists():
        raise FileNotFoundError(f"HarpSync file not found: {harpsync_path}")

    df = pd.read_csv(harpsync_path)
    required = {"Seconds", "Value.Clock", "Value.HubClock", "Value.HarpTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HarpSync file missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Estimate actual clock rate from the data (more accurate than nominal)
    hub_diffs = np.diff(df["Value.HubClock"].values)
    sec_diffs = np.diff(df["Seconds"].values)
    estimated_hz = float((hub_diffs / sec_diffs).mean())

    print(f"HarpSync loaded: {len(df)} rows  "
          f"({df['Seconds'].iloc[-1] - df['Seconds'].iloc[0]} s)")
    print(f"  Estimated HubClock rate : {estimated_hz:,.1f} Hz  "
          f"(nominal {ONIX_CLOCK_HZ_NOMINAL:,.0f} Hz, "
          f"error {abs(estimated_hz - ONIX_CLOCK_HZ_NOMINAL)/ONIX_CLOCK_HZ_NOMINAL*100:.3f}%)")
    print(f"  HubClock jitter         : "
          f"{hub_diffs.std() / estimated_hz * 1e6:.3f} µs per second")

    df.attrs["estimated_hub_clock_hz"] = estimated_hz
    return df


def ap_samples_to_harp_time(
    sample_indices: np.ndarray,
    harpsync: "pd.DataFrame",
    ap_clock: Optional[np.ndarray] = None,
    ap_sample0_hub_clock: Optional[int] = None,
    sampling_rate: float = 30_000.0,
) -> np.ndarray:
    """
    Convert AP sample indices to Harp time (seconds) using HarpSync anchors.

    Two modes depending on what data you have:

    Mode A — Clock file available (preferred, accurate to <10 µs):
        Pass the full ap_clock array (from load_ap_timestamps()).
        The actual HubClock count at each sample is used directly — no
        fixed-rate assumption, so clock jitter is fully corrected.

    Mode B — Clock file missing (fallback, ~1 s error over 24 hr):
        Pass ap_sample0_hub_clock only. A fixed ticks-per-sample rate
        is assumed, which accumulates ~3.6 ms error per 5 minutes and
        grows to ~1 s over a 24 hr recording.

    Parameters
    ----------
    sample_indices
        Array of AP sample indices (integers), e.g. spike times from KS.
    harpsync
        DataFrame returned by load_harpsync().
    ap_clock
        Full Clock array from load_ap_timestamps() — uint64, one value
        per AP sample. If provided, ap_sample0_hub_clock is ignored.
    ap_sample0_hub_clock
        HubClock count at sample 0. Used only when ap_clock is None.
    sampling_rate
        AP sampling rate in Hz (30 000 for NP2.0 via ONIX).

    Returns
    -------
    np.ndarray of float64 — Harp time in seconds for each sample index.
    """
    anchor_hub  = harpsync["Value.HubClock"].values.astype(np.float64)
    anchor_harp = harpsync["Value.HarpTime"].values.astype(np.float64)

    if ap_clock is not None:
        # Mode A: look up the actual HubClock count for each sample index
        hub_clocks = ap_clock[sample_indices].astype(np.float64)
        mode = "A (Clock file — accurate)"
    elif ap_sample0_hub_clock is not None:
        # Mode B: assume fixed rate from sample 0
        estimated_hz = harpsync.attrs.get("estimated_hub_clock_hz", ONIX_CLOCK_HZ_NOMINAL)
        hub_clocks = (ap_sample0_hub_clock
                      + sample_indices.astype(np.float64) * (estimated_hz / sampling_rate))
        mode = "B (fixed rate — ~1 s error over 24 hr)"
    else:
        raise ValueError(
            "Provide either ap_clock (preferred) or ap_sample0_hub_clock (fallback)."
        )

    harp_times = np.interp(hub_clocks, anchor_hub, anchor_harp)
    return harp_times


def estimate_ap_sample0_hub_clock(
    harpsync: "pd.DataFrame",
    session_start_harp_time: float,
    sampling_rate: float = 30_000.0,
) -> int:
    """
    Estimate the HubClock count at AP sample 0 when no per-sample
    timestamp file is available.

    Uses the HarpSync anchor point closest to session_start_harp_time
    and extrapolates to t=0.

    Parameters
    ----------
    session_start_harp_time
        Harp time (seconds) at which the AP recording started.
        Typically the Value.HarpTime of the first HarpSync row, or
        a known session start time from your Bonsai workflow log.
    """
    estimated_hz = harpsync.attrs.get("estimated_hub_clock_hz", ONIX_CLOCK_HZ_NOMINAL)

    # Find the anchor closest to the start time
    idx = np.argmin(np.abs(harpsync["Value.HarpTime"].values - session_start_harp_time))
    anchor_hub  = harpsync["Value.HubClock"].iloc[idx]
    anchor_harp = harpsync["Value.HarpTime"].iloc[idx]

    # Extrapolate back to session_start_harp_time
    delta_s   = session_start_harp_time - anchor_harp
    delta_hub = delta_s * estimated_hz
    sample0_hub = int(anchor_hub + delta_hub)

    print(f"Estimated sample-0 HubClock: {sample0_hub:,}  "
          f"(from anchor at HarpTime {anchor_harp}, "
          f"Δ = {delta_s:.3f} s)")
    return sample0_hub


# ------------------------------------------------------------
# Per-sample AP timestamp loading (if present)
# ------------------------------------------------------------

def load_ap_timestamps(
    raw_dir: str | Path,
    probe_prefix: str = "NeuropixelsV2_ProbeA",
) -> Optional[np.ndarray]:
    """
    Load per-sample ONIX HubClock counts from the AP Clock binary files.

    bonsai-onix1 saves a companion clock file alongside each AP .bin chunk,
    named e.g.:

        NeuropixelsV2_ProbeA_Clock_0.bin

    Each file is a flat array of uint64 values — one HubClock count per AP
    sample, in the same order as the corresponding AmplifierData .bin chunk.
    The clock runs at ~249.42 MHz (verify from HarpSync).

    Parameters
    ----------
    raw_dir
        Folder containing the Clock .bin files.
    probe_prefix
        Filename prefix before "_Clock_N.bin", e.g. "NeuropixelsV2_ProbeA".

    Returns
    -------
    np.ndarray of uint64, shape (n_samples,), or None if no files found.
    """
    raw_dir = Path(raw_dir)

    clock_files = sorted(
        raw_dir.glob(f"{probe_prefix}_Clock_*.bin"),
        key=_chunk_index,
    )
    if not clock_files:
        print("No per-sample Clock .bin files found. "
              "Use HarpSync + estimate_ap_sample0_hub_clock() instead.")
        return None

    all_counts = []
    for cf in clock_files:
        counts = np.fromfile(cf, dtype=np.uint64)
        all_counts.append(counts)
        dur_s = len(counts) / 30_000
        print(f"  {cf.name}: {len(counts):,} samples  ({dur_s:.1f} s)")

    counts_cat = np.concatenate(all_counts)

    # Sanity check: diffs should be ~8314 ticks (249.42 MHz / 30 kHz)
    diffs = np.diff(counts_cat[:10_000])
    mean_diff = float(diffs.mean())
    implied_hz = mean_diff * 30_000
    print(f"\nClock file loaded: {len(counts_cat):,} samples total")
    print(f"  Implied clock rate : {implied_hz:,.0f} Hz  "
          f"(expected ~249,423,527 Hz)")
    print(f"  Sample-0 HubClock  : {counts_cat[0]:,}  <- use as ap_sample0_hub_clock")

    return counts_cat


# Helpers

def _chunk_index(path: Path) -> int:
    """Extract trailing integer from filename for sorting, e.g. _0.bin → 0."""
    m = re.search(r"_(\d+)(?:_timestamps)?\.(?:bin|csv)$", path.name)
    return int(m.group(1)) if m else 0

def _build_np2_fallback_probe(probe_name: str, n_channels: int):
    """Build a simple local NP2 geometry when online probe download is unavailable."""
    from probeinterface import Probe

    if probe_name == "Neuropixels2.0-1shank":
        if n_channels != 384:
            raise ValueError(
                "Fallback geometry for Neuropixels2.0-1shank expects n_channels=384. "
                f"Received n_channels={n_channels}."
            )
        n_rows = n_channels // 2
        x = np.tile(np.array([0.0, 32.0]), n_rows)
        y = np.repeat(np.arange(n_rows, dtype=float) * 15.0, 2)
        positions = np.c_[x, y]

    elif probe_name == "Neuropixels2.0-4shank":
        if n_channels % 4 != 0:
            raise ValueError(
                "Fallback geometry for Neuropixels2.0-4shank expects n_channels divisible by 4. "
                f"Received n_channels={n_channels}."
            )
        per_shank = n_channels // 4
        if per_shank % 2 != 0:
            raise ValueError(
                "Fallback geometry for Neuropixels2.0-4shank expects an even channel count per shank. "
                f"Received per_shank={per_shank}."
            )

        shank_spacing_um = 250.0
        n_rows = per_shank // 2
        shank_positions = []
        for shank_idx in range(4):
            x0 = shank_idx * shank_spacing_um
            x = np.tile(np.array([x0, x0 + 32.0]), n_rows)
            y = np.repeat(np.arange(n_rows, dtype=float) * 15.0, 2)
            shank_positions.append(np.c_[x, y])
        positions = np.vstack(shank_positions)

    else:
        raise ValueError(
            "Could not load probe geometry online and no fallback geometry is "
            f"implemented for probe_name='{probe_name}'."
        )

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shape_params={"width": 12.0},
    )
    return probe

def _chunk_index(path: Path) -> int:
    """Extract trailing integer from filename for sorting, e.g. _0.bin → 0."""
    m = re.search(r"_(\d+)(?:_timestamps)?\.(?:bin|csv)$", path.name)
    return int(m.group(1)) if m else 0
