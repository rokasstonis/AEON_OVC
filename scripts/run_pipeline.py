"""
scripts/run_pipeline.py
-----------------------
End-to-end ONIX NP2.0 preprocessing and spike sorting pipeline.

Usage
-----
    python scripts/run_pipeline.py --config config.yaml [--session SESSION_ID]

Steps
-----
1.  Load ONIX binary + attach probe geometry
2.  Build lazy preprocessing chain (filter → phase shift → bad channels → CAR)
3.  Inspect for drift (optional, saves a figure)
4.  Save preprocessed binary to data/processed/
5.  Run spike sorter
6.  Post-process + compute quality metrics
7.  Export sorting analyser to data/sorted/

After this script completes, run Phy for manual curation:
    phy template-gui data/sorted/<session>/sorter_output/params.py
"""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spikeinterface.full as si
import importlib.util

# # Add src/ to path so we can import np2_pipeline without installing
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# from np2_pipeline.io import load_onix_recording, load_onix_timestamps
# from np2_pipeline.preprocess import build_preprocessing_chain, check_drift

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


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="NP2.0 ONIX pipeline")
    parser.add_argument("--config", default="config.yml",
                        help="Path to config YML file")
    parser.add_argument("--session", default=None,
                        help="Override session_id in config")
    parser.add_argument("--skip-sorting", action="store_true",
                        help="Stop after preprocessing (useful for QC)")
    parser.add_argument("--skip-drift-check", action="store_true",
                        help="Skip drift estimation plot")
    return parser.parse_args()


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(cfg: dict, args):

    session_id = args.session or cfg["session"]["session_id"]
    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"{'='*60}\n")

    raw_dir      = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"]) / session_id
    sorted_dir   = Path(cfg["paths"]["sorted_dir"]) / session_id
    results_dir  = Path(cfg["paths"]["results_dir"]) / session_id
    for d in [processed_dir, sorted_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    acq  = cfg["acquisition"]
    prb  = cfg["probe"]
    pre  = cfg["preprocessing"]
    srt  = cfg["sorting"]
    qm   = cfg["quality_metrics"]

    # ------------------------------------------------------------------
    # 1. Load raw recording
    # ------------------------------------------------------------------
    print("\n--- Step 1: Load raw data ---")
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

    # Optionally load timestamps for behavioural alignment
    try:
        times = load_ap_timestamps(raw_dir)
        np.save(processed_dir / "hardware_timestamps.npy", times)
        print(f"Hardware timestamps saved to {processed_dir}/hardware_timestamps.npy")
    except FileNotFoundError as e:
        print(f"Warning: timestamps not found ({e})")

    # ------------------------------------------------------------------
    # 2. Build preprocessing chain (lazy)
    # ------------------------------------------------------------------
    print("\n--- Step 2: Preprocessing chain ---")
    rec_preprocessed, bad_channels = build_preprocessing_chain(
    recording,
    highpass_freq_min=pre['highpass_freq_min'],
    bad_channel_score_threshold=pre['bad_channel_score_threshold'],
    artifact_removal=pre['artifact_removal'])

    # Save bad channel list for provenance
    with open(processed_dir / "bad_channels.txt", "w") as f:
        f.write("\n".join(map(str, bad_channels)))

    # ------------------------------------------------------------------
    # 3. Drift check (optional)
    # ------------------------------------------------------------------
    if not args.skip_drift_check:
        print("\n--- Step 3: Drift estimation ---")
        peaks, locs = check_drift(
            rec_preprocessed,
            n_jobs=srt["n_jobs"],
            chunk_duration=srt["chunk_duration"],
        )
        _plot_drift(peaks, locs, recording, results_dir / "drift_map.png")
    else:
        print("\n--- Step 3: Drift check skipped ---")

    # ------------------------------------------------------------------
    # 4. Save preprocessed binary
    # ------------------------------------------------------------------
    print("\n--- Step 4: Save preprocessed binary ---")
    job_kwargs = dict(n_jobs=srt["n_jobs"], chunk_duration=srt["chunk_duration"],
                      progress_bar=True)
    preprocessed_path = processed_dir / "preprocessed_binary"
    if not preprocessed_path.exists():
        rec_preprocessed = rec_preprocessed.save(
            folder=str(preprocessed_path),
            format="binary",
            **job_kwargs,
        )
        print(f"Saved to {preprocessed_path}")
    else:
        print(f"Preprocessed binary already exists at {preprocessed_path}, loading.")
        rec_preprocessed = si.load_extractor(str(preprocessed_path))

    if args.skip_sorting:
        print("\nSkip-sorting flag set — stopping after preprocessing.")
        return

    # ------------------------------------------------------------------
    # 5. Run spike sorter
    # ------------------------------------------------------------------
    print(f"\n--- Step 5: Spike sorting ({srt['sorter']}) ---")
    sorter_output_dir = sorted_dir / srt["sorter"]
    sorting = si.run_sorter(
        srt["sorter"],
        recording=rec_preprocessed,
        folder=str(sorter_output_dir),
        verbose=True,
        **srt.get("sorter_kwargs", {}),
    )
    print(f"Sorting done: {len(sorting.unit_ids)} units found")

    # ------------------------------------------------------------------
    # 6. Post-processing + quality metrics
    # ------------------------------------------------------------------
    print("\n--- Step 6: Post-processing and quality metrics ---")
    analyzer = si.create_sorting_analyzer(
        sorting,
        rec_preprocessed,
        format="binary_folder",
        folder=str(sorted_dir / "sorting_analyzer"),
        sparse=True,
        overwrite=False,
    )
    analyzer.compute([
        "random_spikes",
        "waveforms",
        "templates",
        "noise_levels",
        "spike_amplitudes",
        "correlograms",
        "unit_locations",
        "template_similarity",
    ], **job_kwargs)

    qm_df = si.compute_quality_metrics(
        analyzer,
        metric_names=qm["metrics"],
    )
    qm_df.to_csv(results_dir / "quality_metrics.csv")
    print(f"Quality metrics saved to {results_dir}/quality_metrics.csv")
    print(qm_df.describe())

    print(f"\nPipeline complete for session {session_id}.")
    print(f"To run manual curation:\n"
          f"  phy template-gui {sorter_output_dir}/sorter_output/params.py")


# ============================================================
# Helpers
# ============================================================

def _plot_drift(peaks, locations, recording, save_path: Path):
    """Depth-vs-time scatter plot for drift inspection."""
    times = peaks["sample_index"] / recording.get_sampling_frequency()
    depths = locations["y"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(times, depths, s=0.3, alpha=0.3, c="steelblue", rasterized=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth along probe (µm)")
    ax.set_title("Peak depth vs time — drift inspection")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Drift map saved to {save_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_pipeline(cfg, args)
