"""
scripts/hpc_pipeline.py
-----------------------
HPC-adapted pipeline for 24hr+ NP2.0 ONIX recordings on Ceph-mounted storage.

Key differences from scripts/run_pipeline.py:
  - All intermediate files (preprocessed binary, KS temp) go to $TMPDIR or
    a user-specified scratch path, NOT to Ceph. Ceph is read for raw data
    and written to only at the end (results + SortingAnalyzer).
  - Uses Singularity instead of Docker for KS2.5 (Docker unavailable on HPC).
  - Parallelism scaled to the job allocation (n_jobs from env var / config).
  - Progress is written to a log file suitable for inspection via `tail -f`.

Usage (from job script — do not call directly)
-----
    python scripts/hpc_pipeline.py \
        --config config.yml \
        --scratch /scratch/scratch/zcqsrxxx/np2_session \
        --ceph-results /ceph/neuroinformatics/rokas/results/session_id
"""

from __future__ import annotations
import argparse
import importlib.util
import logging
import os
import shutil
import sys
import traceback
import time
from pathlib import Path

import yaml
import numpy as np
import spikeinterface.full as si

# Make the repository's src/ directory importable when this file is run as a script.
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

def _load_local_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Module file does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    io_path = repo_root / "src" / "io.py"
    custom_io = _load_local_module("custom_io", io_path)

    preprocess_path = repo_root / "src" / "preprocess.py"
    custom_preprocess = _load_local_module("custom_preprocess", preprocess_path)

    if not hasattr(custom_io, "load_onix_recording"):
        raise AttributeError("Missing symbol in io.py: load_onix_recording")
    if not hasattr(custom_io, "load_ap_timestamps"):
        raise AttributeError("Missing symbol in io.py: load_ap_timestamps")
    if not hasattr(custom_preprocess, "build_preprocessing_chain"):
        raise AttributeError("Missing symbol in preprocess.py: build_preprocessing_chain")
    if not hasattr(custom_preprocess, "check_drift"):
        raise AttributeError("Missing symbol in preprocess.py: check_drift")

    load_onix_recording = custom_io.load_onix_recording
    load_ap_timestamps = custom_io.load_ap_timestamps
    build_preprocessing_chain = custom_preprocess.build_preprocessing_chain
    check_drift = custom_preprocess.check_drift
except Exception:
    print("[FATAL] Failed to load local pipeline modules.", file=sys.stderr)
    print(f"  repo_root: {repo_root}", file=sys.stderr)
    print(f"  expected io.py: {repo_root / 'src' / 'io.py'}", file=sys.stderr)
    print(f"  expected preprocess.py: {repo_root / 'src' / 'preprocess.py'}", file=sys.stderr)
    traceback.print_exc()
    raise


# ============================================================
# Logging — write to both stdout and file
# ============================================================

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

log = logging.getLogger(__name__)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yml")
    p.add_argument("--scratch", required=True,
                   help="Fast local scratch directory (e.g. $TMPDIR or /scratch/…)")
    p.add_argument("--ceph-results", required=True,
                   help="Ceph path to write final results/analyzer back to")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Override n_jobs (defaults to SLURM_CPUS_PER_TASK env var)")
    p.add_argument("--skip-sorting", action="store_true")
    p.add_argument("--skip-drift-check", action="store_true")
    return p.parse_args()


# ============================================================
# Pipeline
# ============================================================

def run_hpc_pipeline(cfg: dict, args):

    scratch      = Path(args.scratch)
    ceph_results = Path(args.ceph_results)
    session_id   = cfg["session"]["session_id"]

    # Work entirely on scratch during the job
    scratch_processed = scratch / "processed"
    scratch_sorted    = scratch / "sorted"
    for d in [scratch_processed, scratch_sorted, ceph_results]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = scratch / "pipeline.log"
    setup_logging(log_path)

    log.info("=" * 60)
    log.info(f"Session : {session_id}")
    log.info(f"Scratch : {scratch}")
    log.info(f"Results → Ceph : {ceph_results}")
    log.info("=" * 60)

    acq = cfg["acquisition"]
    prb = cfg["probe"]
    pre = cfg["preprocessing"]
    srt = cfg["sorting"]
    qm  = cfg["quality_metrics"]

    # Resolve n_jobs: CLI > env var > config
    n_jobs = (
        args.n_jobs
        or int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        or int(os.environ.get("NSLOTS", 0))   # SGE equivalent
        or cfg["sorting"]["n_jobs"]
    )
    log.info(f"n_jobs = {n_jobs}")

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration=srt["chunk_duration"], progress_bar=True)

    raw_dir = Path(cfg["paths"]["raw_dir"])  # Ceph path to raw data

    # ----------------------------------------------------------------
    # 1. Load raw recording from Ceph (lazy — nothing read yet)
    # ----------------------------------------------------------------
    log.info("Step 1: Load raw recording (lazy)")
    t0 = time.time()

    recording = load_onix_recording(
        raw_dir=raw_dir,
        probe_name=prb["model"],
        manufacturer=prb["manufacturer"],
        sampling_frequency=acq["sampling_frequency"],
        n_channels=acq["n_channels"],
        dtype=acq["dtype"],
        gain_to_uV=acq["gain_to_uV"],
        offset_to_uV=acq["offset_to_uV"],
        probeinterface_json=prb.get("probeinterface_json"),
    )

    dur_hr = recording.get_total_duration() / 3600
    log.info(f"  Duration: {dur_hr:.2f} hr  |  {recording.get_num_channels()} channels")

    # Save timestamps to scratch (fast), copy to Ceph at end
    try:
        times = load_ap_timestamps(raw_dir)
        np.save(scratch_processed / "hardware_timestamps.npy", times)
        log.info("  Hardware timestamps saved to scratch")
    except FileNotFoundError:
        log.warning("  No timestamp files — behavioural alignment unavailable")

    # ----------------------------------------------------------------
    # 2. Build preprocessing chain (still lazy)
    # ----------------------------------------------------------------
    log.info("Step 2: Building preprocessing chain")
    rec_pre, bad_channels = build_preprocessing_chain(
        recording,
        highpass_freq_min=pre["highpass_freq_min"],
        bad_channel_score_threshold=pre["bad_channel_score_threshold"],
        artifact_removal=pre["artifact_removal"],
        car_local_inner_radius_um=pre.get("car_local_inner_radius_um", 50),
        car_local_outer_radius_um=pre.get("car_local_outer_radius_um", 100),
        verbose=False,
    )
    log.info(f"  Bad channels: {len(bad_channels)} — {bad_channels}")
    with open(scratch_processed / "bad_channels.txt", "w") as f:
        f.write("\n".join(map(str, bad_channels)))

    # ----------------------------------------------------------------
    # 3. Drift check (lightweight — subset of peaks)
    # ----------------------------------------------------------------
    if not args.skip_drift_check:
        log.info("Step 3: Drift estimation")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # For 24hr recordings, use only the first and last 30 min + middle 30 min
        # to get a fast drift overview without processing the full file
        n_total   = recording.get_num_samples()
        fs        = recording.get_sampling_frequency()
        n_30min   = int(30 * 60 * fs)

        segments  = [
            (0, n_30min),
            (n_total // 2 - n_30min // 2, n_total // 2 + n_30min // 2),
            (max(0, n_total - n_30min), n_total),
        ]
        seg_labels = ["first 30 min", "middle 30 min", "last 30 min"]
        all_peaks, all_locs, all_labels = [], [], []

        for (s_start, s_end), label in zip(segments, seg_labels):
            rec_seg = rec_pre.frame_slice(s_start, s_end)
            peaks, locs = check_drift(rec_seg, n_jobs=n_jobs, chunk_duration=srt["chunk_duration"])
            # Shift sample indices back to absolute time
            times_seg = (peaks["sample_index"] + s_start) / fs
            all_peaks.append(times_seg)
            all_locs.append(locs["y"])
            all_labels.append(label)
            log.info(f"  {label}: {len(peaks):,} peaks")

        fig, ax = plt.subplots(figsize=(14, 5))
        colours = ["steelblue", "darkorange", "forestgreen"]
        for t, y, lbl, c in zip(all_peaks, all_locs, all_labels, colours):
            ax.scatter(t / 3600, y, s=0.3, alpha=0.4, c=c, label=lbl, rasterized=True)
        ax.set_xlabel("Time (hr)"); ax.set_ylabel("Depth (µm)")
        ax.set_title("Drift overview — sampled segments")
        ax.legend(markerscale=8)
        fig.tight_layout()
        fig.savefig(scratch / "drift_overview.png", dpi=150)
        plt.close(fig)
        log.info(f"  Drift plot saved to {scratch}/drift_overview.png")
    else:
        log.info("Step 3: Drift check skipped")

    # ----------------------------------------------------------------
    # 4. Save preprocessed binary to SCRATCH (fast storage for KS)
    # ----------------------------------------------------------------
    preproc_path = scratch_processed / "preprocessed_binary"
    if preproc_path.exists():
        log.info(f"Step 4: Preprocessed binary exists on scratch, loading")
        rec_pre = si.load_extractor(str(preproc_path))
    else:
        log.info("Step 4: Writing preprocessed binary to scratch")
        log.info("  (This streams directly from Ceph → scratch: expect ~1 hr/TB)")
        t_write = time.time()
        rec_pre = rec_pre.save(
            folder=str(preproc_path),
            format="binary",
            **job_kwargs,
        )
        log.info(f"  Done in {(time.time() - t_write)/60:.1f} min")

    if args.skip_sorting:
        log.info("skip-sorting flag set — stopping after preprocessing")
        _copy_results_to_ceph(scratch, ceph_results, log)
        return

    # ----------------------------------------------------------------
    # 5. Spike sorting (reading from scratch, writing to scratch)
    # ----------------------------------------------------------------
    log.info(f"Step 5: Spike sorting — {srt['sorter']}")
    sorter_output = scratch_sorted / srt["sorter"]

    # Kilosort 2.5 via Singularity (HPC doesn't have Docker)
    # Kilosort 4 can run natively (installed in conda env)
    sorter_kwargs = dict(srt.get("sorter_kwargs", {}))

    if srt["sorter"] == "kilosort2_5":
        singularity_image = cfg.get("hpc", {}).get(
            "singularity_image",
            "docker://spikeinterface/kilosort2_5-compiled-base"
        )
        log.info(f"  Using Singularity image: {singularity_image}")
        sorting = si.run_sorter(
            "kilosort2_5",
            recording=rec_pre,
            folder=str(sorter_output),
            singularity_image=singularity_image,
            verbose=True,
            **sorter_kwargs,
        )
    else:
        # KS4, SpykingCircus2, etc. — run natively
        sorting = si.run_sorter(
            srt["sorter"],
            recording=rec_pre,
            folder=str(sorter_output),
            verbose=True,
            **sorter_kwargs,
        )

    n_units = len(sorting.unit_ids)
    log.info(f"  Sorting done: {n_units} units")

    # ----------------------------------------------------------------
    # 6. Post-processing + quality metrics (on scratch)
    # ----------------------------------------------------------------
    log.info("Step 6: Post-processing and quality metrics")
    analyzer = si.create_sorting_analyzer(
        sorting,
        rec_pre,
        format="binary_folder",
        folder=str(scratch_sorted / "sorting_analyzer"),
        sparse=True,
        overwrite=False,
    )
    analyzer.compute([
        "random_spikes", "waveforms", "templates", "noise_levels",
        "spike_amplitudes", "correlograms", "unit_locations",
        "template_similarity",
    ], **job_kwargs)

    qm_df = si.compute_quality_metrics(analyzer, metric_names=qm["metrics"])
    qm_df.to_csv(scratch / "quality_metrics.csv")
    log.info(f"  {n_units} units | QC metrics saved")
    log.info(f"\n{qm_df.describe().to_string()}")

    # ----------------------------------------------------------------
    # 7. Copy results back to Ceph (only the small stuff)
    # ----------------------------------------------------------------
    _copy_results_to_ceph(scratch, ceph_results, log)

    total_min = (time.time() - t0) / 60
    log.info(f"\nPipeline complete in {total_min:.0f} min ({total_min/60:.1f} hr)")
    log.info(f"Phy curation:\n  phy template-gui "
             f"{ceph_results}/sorted/{srt['sorter']}/sorter_output/params.py")


def _copy_results_to_ceph(scratch: Path, ceph_results: Path, log):
    """
    Copy lightweight outputs back to Ceph. The preprocessed binary (~2 TB)
    stays on scratch — copy only what you need for analysis.
    """
    log.info("Step 7: Copying results to Ceph")
    copy_items = [
        scratch / "drift_overview.png",
        scratch / "quality_metrics.csv",
        scratch / "pipeline.log",
        scratch / "processed" / "bad_channels.txt",
        scratch / "processed" / "hardware_timestamps.npy",
        scratch / "sorted" / "sorting_analyzer",  # ~15 GB — waveforms etc
    ]
    for item in copy_items:
        if not item.exists():
            log.warning(f"  Not found (skipping): {item.name}")
            continue
        dest = ceph_results / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        log.info(f"  Copied {item.name} → Ceph")

    # Also copy the sorter output folder (params.py + spike_times.npy etc.)
    sorter_out = scratch / "sorted"
    dest_sorted = ceph_results / "sorted"
    if sorter_out.exists():
        if dest_sorted.exists():
            shutil.rmtree(dest_sorted)
        shutil.copytree(sorter_out, dest_sorted)
        log.info(f"  Copied sorted/ → Ceph")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_hpc_pipeline(cfg, args)
