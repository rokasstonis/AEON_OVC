from __future__ import annotations
# import csv
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import spikeinterface.core as si
from probeinterface import get_probe, read_probeinterface



def load_onix_recording(
    raw_dir: str | Path,
    probe_name: str = "Neuropixels2.0-1shank",
    manufacturer: str = "imec",
    sampling_frequency: float = 30000.0,
    n_channels: int = 384,
    dtype: str = "uint16",
    gain_to_uV: float = 2.34375,
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
            # gain_to_uV=gain_to_uV,
            # offset_to_uV=offset_to_uV,
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

    # Summary
    dur = recording.get_total_duration()
    print(f"\nRecording loaded:")
    print(f"  Duration   : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Channels   : {recording.get_num_channels()}")
    print(f"  Sampling   : {recording.get_sampling_frequency()} Hz")
    print(f"  dtype      : {recording.get_dtype()}")
    print(f"  Probe      : {recording.get_probe()}")

    return recording


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
