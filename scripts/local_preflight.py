"""
scripts/local_preflight.py
--------------------------
Run this LOCALLY on the raw data before transferring to HPC or submitting jobs.
It catches the most common failure modes that would otherwise surface
mid-way through a multi-hour compute job.

Usage
-----
    conda activate np2_pipeline
    python scripts/local_preflight.py --config config.yaml

Checks performed
----------------
1.  Binary chunk discovery and size consistency
2.  Estimated total duration vs expected session length
3.  Dtype sanity (are values in a plausible uint16 ADC range?)
4.  Per-channel noise from the first 60 s (fast, catches gain errors)
5.  Timestamp file presence and clock continuity (no dropped samples)
6.  Probe geometry attachment (does SpikeInterface accept the probe?)
7.  Storage space estimate for HPC (preprocessed binary + sorter temp files)
8.  Produces a preflight_report.txt summary you can read before submitting
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import importlib.util

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Make the repository's src/ directory importable when this file is run as a script.
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

io_path = repo_root / "src" / "io.py"
spec = importlib.util.spec_from_file_location('custom_io', io_path)
custom_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_io)
load_onix_recording = custom_io.load_onix_recording
load_ap_timestamps = custom_io.load_ap_timestamps

preprocess_path = repo_root / "src" / "preprocess.py"
preprocess_spec = importlib.util.spec_from_file_location("custom_preprocess", preprocess_path)
custom_preprocess = importlib.util.module_from_spec(preprocess_spec)
preprocess_spec.loader.exec_module(custom_preprocess)
build_preprocessing_chain = custom_preprocess.build_preprocessing_chain


PASS = "  [ PASS ]"
FAIL = "  [ FAIL ]"
WARN = "  [ WARN ]"


def check_chunks(raw_dir: Path, n_channels: int, dtype: str, report: list) -> list[Path]:
    """Verify all .bin chunks are present and internally consistent."""
    print("\n--- 1. Binary chunk discovery ---")
    dtype_bytes = np.dtype(dtype).itemsize

    bin_files = sorted(raw_dir.glob("NeuropixelsV2_Probe*_AmplifierData_*.bin"))
    if not bin_files:
        bin_files = sorted(raw_dir.glob("*.bin"))

    if not bin_files:
        msg = f"{FAIL} No .bin files found in {raw_dir}"
        print(msg); report.append(msg)
        return []

    total_bytes = 0
    issues = []
    for bf in bin_files:
        size = bf.stat().st_size
        total_bytes += size
        remainder = size % (n_channels * dtype_bytes)
        if remainder != 0:
            issues.append(f"     {bf.name}: size {size} not divisible by "
                          f"{n_channels * dtype_bytes} (n_channels × dtype_bytes) "
                          f"— last chunk may be truncated")
        else:
            n_samples = size // (n_channels * dtype_bytes)
            print(f"  {bf.name}: {size/1e9:.2f} GB  ({n_samples:,} samples)")

    if issues:
        for iss in issues:
            print(f"{WARN}{iss}")
            report.append(f"{WARN} Chunk size warning: {iss}")
    else:
        msg = f"{PASS} {len(bin_files)} chunk(s) found, all sizes consistent"
        print(msg); report.append(msg)

    total_samples = total_bytes // (n_channels * dtype_bytes)
    return bin_files, total_samples, total_bytes


def check_duration(total_samples: int, sampling_frequency: float, report: list):
    """Check that total duration is plausible."""
    print("\n--- 2. Total duration ---")
    duration_s  = total_samples / sampling_frequency
    duration_hr = duration_s / 3600

    print(f"  Total samples : {total_samples:,}")
    print(f"  Duration      : {duration_s:,.0f} s  ({duration_hr:.2f} hr)")

    if duration_hr < 0.1:
        msg = f"{WARN} Very short recording ({duration_hr:.2f} hr) — is this the full session?"
        print(msg); report.append(msg)
    elif duration_hr > 30:
        msg = f"{WARN} Very long recording ({duration_hr:.2f} hr) — verify this is correct"
        print(msg); report.append(msg)
    else:
        msg = f"{PASS} Duration {duration_hr:.2f} hr looks plausible"
        print(msg); report.append(msg)

    return duration_s


def check_dtype_range(bin_files: list[Path], n_channels: int, dtype: str,
                      sampling_frequency: float, report: list):
    """
    Read 10 s from the middle of the first chunk and check ADC values.
    uint16 from NP2.0 should span a range centred on ~32768 (mid-scale),
    typically 30000–35000 for in-vivo data.
    """
    print("\n--- 3. Dtype and value range sanity ---")
    n_check = int(10 * sampling_frequency)  # 10 s
    dt      = np.dtype(dtype)

    bf = bin_files[len(bin_files) // 2]  # middle chunk
    file_samples = bf.stat().st_size // (n_channels * dt.itemsize)
    start = max(0, file_samples // 2 - n_check // 2)
    byte_offset = start * n_channels * dt.itemsize

    data = np.memmap(bf, dtype=dt, mode="r",
                     offset=byte_offset,
                     shape=(n_check, n_channels))

    vmin, vmax = int(data.min()), int(data.max())
    vmean = float(data.mean())
    vstd  = float(data.std())
    print(f"  Sampled 10 s from middle of {bf.name}")
    print(f"  Value range : {vmin} – {vmax}  (uint16 full range: 0–65535)")
    print(f"  Mean ± std  : {vmean:.0f} ± {vstd:.0f}")

    issues = []
    if dtype == "uint16":
        if vmean < 20000 or vmean > 45000:
            issues.append(f"Mean {vmean:.0f} is far from mid-scale (32768) — "
                          "possible gain misconfiguration or wrong dtype")
        if vstd < 500:
            issues.append(f"Very low std ({vstd:.0f}) — possible dead probe or wrong file")
        if vmax == 65535 or vmin == 0:
            issues.append("Data is saturated (values at ADC limits) — "
                          "check headstage power / grounding")

    if issues:
        for iss in issues:
            msg = f"{WARN} {iss}"
            print(msg); report.append(msg)
    else:
        msg = f"{PASS} ADC values look healthy (mean ≈ {vmean:.0f}, std ≈ {vstd:.0f})"
        print(msg); report.append(msg)

    del data


def check_noise(raw_dir: Path, n_channels: int, dtype: str,
                sampling_frequency: float, gain_to_uV: float,
                results_dir: Path, report: list):
    """
    Estimate per-channel noise from the first 60 s.
    Expected in-vivo NP2.0: ~10–25 µV RMS.
    """
    print("\n--- 4. Per-channel noise estimate (first 60 s) ---")
    import spikeinterface.full as si

    try:
        recording = load_onix_recording(
            raw_dir=raw_dir,
            gain_to_uV=gain_to_uV,
            offset_to_uV=0.0,
            sampling_frequency=sampling_frequency,
            n_channels=n_channels,
            dtype=dtype,
        )
    except Exception as e:
        msg = f"{FAIL} Could not load recording: {e}"
        print(msg); report.append(msg)
        return

    # Grab first 60 s for noise estimate
    n_60s = min(int(60 * sampling_frequency), recording.get_num_samples())
    rec_sub = recording.frame_slice(0, n_60s)
    rec_hp = si.highpass_filter(rec_sub, freq_min=300)

    noise = si.get_noise_levels(rec_hp, method="std", return_in_uV=True)

    med_noise = float(np.median(noise))
    n_bad_lo  = int(np.sum(noise < 3))
    n_bad_hi  = int(np.sum(noise > 50))

    print(f"  Median noise : {med_noise:.1f} µV RMS")
    print(f"  Channels < 3 µV (likely dead) : {n_bad_lo}")
    print(f"  Channels > 50 µV (noisy)      : {n_bad_hi}")

    # Save noise figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(noise, lw=0.7, color="steelblue")
    ax.axhline(med_noise, color="r", ls="--",
               label=f"median = {med_noise:.1f} µV")
    ax.axhspan(0, 3,  alpha=0.15, color="red",   label="likely dead (< 3 µV)")
    ax.axhspan(50, max(noise.max(), 55), alpha=0.15, color="orange",
               label="noisy (> 50 µV)")
    ax.set_xlabel("Channel index"); ax.set_ylabel("Noise RMS (µV)")
    ax.set_title("Per-channel noise — first 60 s (high-pass filtered)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    noise_fig = results_dir / "preflight_noise.png"
    fig.savefig(noise_fig, dpi=150); plt.close(fig)
    print(f"  Noise figure saved → {noise_fig}")

    if med_noise < 5 or med_noise > 40:
        msg = (f"{WARN} Median noise {med_noise:.1f} µV outside expected range "
               f"(10–25 µV). Check gain_to_uV setting and grounding.")
        print(msg); report.append(msg)
    elif n_bad_lo > 10 or n_bad_hi > 20:
        msg = (f"{WARN} {n_bad_lo} dead + {n_bad_hi} noisy channels — "
               f"bad channel interpolation will handle these but worth inspecting")
        print(msg); report.append(msg)
    else:
        msg = (f"{PASS} Noise looks healthy: median {med_noise:.1f} µV RMS, "
               f"{n_bad_lo} dead, {n_bad_hi} noisy")
        print(msg); report.append(msg)


def check_timestamps(raw_dir: Path, sampling_frequency: float, report: list):
    """Check timestamp files exist and the clock is continuous (no dropped frames)."""
    print("\n--- 5. Timestamp continuity ---")

    times = load_ap_timestamps(raw_dir)
    if times is None:
        msg = f"{WARN} No per-sample AP clock files found. Behavioural alignment won't be possible."
        print(msg); report.append(msg)
        return

    diffs_us = np.diff(times) * 1e6
    expected_us = 1e6 / sampling_frequency    # ~33.33 µs at 30 kHz

    # A "gap" is a diff more than 2× the expected interval
    gaps = np.where(diffs_us > 2 * expected_us)[0]
    jitter_us = float(np.std(diffs_us))

    print(f"  Total samples   : {len(times):,}")
    print(f"  Expected ISI    : {expected_us:.2f} µs")
    print(f"  Jitter (std)    : {jitter_us:.3f} µs")
    print(f"  Gaps detected   : {len(gaps)}")
    if len(gaps):
        for g in gaps[:5]:
            print(f"    Sample {g:,}: gap = {diffs_us[g]:.1f} µs")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more")

    if len(gaps) > 0:
        msg = (f"{WARN} {len(gaps)} timestamp gap(s) detected — "
               f"dropped samples or PCIe buffer overflow. Check ONIX log.")
        print(msg); report.append(msg)
    elif jitter_us > 5:
        msg = f"{WARN} High timestamp jitter ({jitter_us:.2f} µs) — ONIX clock issue?"
        print(msg); report.append(msg)
    else:
        msg = f"{PASS} Timestamps continuous, jitter {jitter_us:.3f} µs"
        print(msg); report.append(msg)


def check_probe_attachment(raw_dir: Path, probe_model: str,
                           probeinterface_json, report: list):
    """Verify SpikeInterface accepts the probe geometry without error."""
    print("\n--- 6. Probe geometry attachment ---")
    import spikeinterface.core as si

    try:
        recording = load_onix_recording(
            raw_dir=raw_dir,
            probe_name=probe_model,
            probeinterface_json=probeinterface_json,
        )
        probe = recording.get_probe()
        n_contacts = probe.get_contact_count()
        msg = (f"{PASS} Probe attached: {probe_model}, "
               f"{n_contacts} contacts, "
               f"depth range {probe.contact_positions[:, 1].min():.0f}–"
               f"{probe.contact_positions[:, 1].max():.0f} µm")
        print(msg); report.append(msg)
    except Exception as e:
        msg = f"{FAIL} Probe attachment failed: {e}"
        print(msg); report.append(msg)


def check_storage_estimate(total_bytes: int, duration_s: float, report: list):
    """
    Estimate disk space needed on HPC scratch for the full pipeline.

    Preprocessed binary  : ~same size as raw (float32 instead of uint16 → ×2)
    Kilosort temp files  : ~1–3× raw size during sorting
    SortingAnalyzer      : ~5–20 GB (waveforms, sparse)
    """
    print("\n--- 7. HPC storage estimate ---")

    raw_gb        = total_bytes / 1e9
    preproc_gb    = raw_gb * 2       # uint16 → float32 after preprocessing save
    ks_scratch_gb = raw_gb * 2       # KS temp files (conservative)
    analyzer_gb   = 15               # waveforms (sparse NP2.0 estimate)
    total_scratch = preproc_gb + ks_scratch_gb + analyzer_gb

    print(f"  Raw data           : {raw_gb:.1f} GB")
    print(f"  Preprocessed binary: {preproc_gb:.1f} GB  (float32)")
    print(f"  KS temp files      : {ks_scratch_gb:.1f} GB  (estimate)")
    print(f"  SortingAnalyzer    : {analyzer_gb:.0f} GB  (sparse waveforms)")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL scratch need : {total_scratch:.0f} GB")
    print()
    print(f"  Tip: preprocess onto scratch, run sorting there, then")
    print(f"  copy only results/ + sorting_analyzer/ back to Ceph.")

    msg = (f"{PASS if total_scratch < 5000 else WARN} "
           f"Estimated scratch needed: {total_scratch:.0f} GB")
    print(msg); report.append(msg)

    if total_scratch > 5000:
        msg = (f"{WARN} > 5 TB scratch needed — check your HPC quota before submitting. "
               f"Consider splitting the session.")
        print(msg); report.append(msg)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Local preflight checks for NP2.0 ONIX data")
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    pre = cfg["preprocessing"]
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    acq = cfg["acquisition"]
    prb = cfg["probe"]

    report = []
    report.append(f"Preflight report — {cfg['session']['session_id']}")
    report.append("=" * 60)

    bin_files, total_samples, total_bytes = check_chunks(
        raw_dir, acq["n_channels"], acq["dtype"], report
    )
    if not bin_files:
        print("\nAborting: no data found.")
        return

    duration_s = check_duration(total_samples, acq["sampling_frequency"], report)
    check_dtype_range(bin_files, acq["n_channels"], acq["dtype"],
                      acq["sampling_frequency"], report)
    check_noise(raw_dir, acq["n_channels"], acq["dtype"],
                acq["sampling_frequency"], acq["gain_to_uV"], results_dir, report)
    check_timestamps(raw_dir, acq["sampling_frequency"], report)
    check_probe_attachment(raw_dir, prb["model"], prb.get("probeinterface_json"), report)
    check_storage_estimate(total_bytes, duration_s, report)

    # Write report
    report_path = results_dir / "preflight_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n{'='*60}")
    print(f"Preflight complete. Report saved to {report_path}")

    n_fail = sum(1 for r in report if "FAIL" in r)
    n_warn = sum(1 for r in report if "WARN" in r)
    print(f"  {n_fail} FAIL  |  {n_warn} WARN  |  "
          f"{len([r for r in report if 'PASS' in r])} PASS")
    if n_fail > 0:
        print("  !! Do not submit HPC jobs until FAIL items are resolved !!")


if __name__ == "__main__":
    main()
