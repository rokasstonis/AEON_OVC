from __future__ import annotations
import spikeinterface.full as si
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing import unsigned_to_signed


def build_preprocessing_chain(
    recording: BaseRecording,
    highpass_freq_min: float = 300.0,
    bad_channel_score_threshold: float = 0.5,
    artifact_removal: str = "car_local",
    car_local_inner_radius_um: float = 50.0,
    car_local_outer_radius_um: float = 100.0,
    verbose: bool = True,
) -> tuple[BaseRecording, list]:
    """
    Apply the full ONIX NP2.0 preprocessing chain and return the lazy recording.

    Steps
    -----
    1. High-pass filter  — isolate AP spike band (≥300 Hz)
    2. Phase shift       — correct inter-sample timing offsets across channels
    3. Bad channel detection + interpolation
    4. Spatial artifact removal (CAR global / CAR local / IBL destripe)

    Parameters
    ----------
    artifact_removal
        "car_global"  : subtract global median across all channels
        "car_local"   : subtract median of a local annulus around each channel
        "destripe"    : IBL spatial high-pass filter (best for quad-shank)

    Returns
    -------
    rec_preprocessed : BaseRecording (lazy)
    bad_channel_ids  : list of str
    """
    if verbose:
        print("=" * 60)
        print("Building preprocessing chain (lazy — no disk writes yet)")
        print("=" * 60)

    # # SpikeInterface preprocessing filters require a signed recording.
    # if recording.get_dtype().kind == "u":
    #     rec = unsigned_to_signed(recording)
    #     if verbose:
    #         print("[0] Converted unsigned recording to signed dtype")
    # else:
    #     rec = recording
        rec = recording
    # ------------------------------------------------------------------
    # Step 1: High-pass filter
    # ------------------------------------------------------------------
    rec = si.highpass_filter(
        rec,
        freq_min=highpass_freq_min,
    )
    if verbose:
        print(f"[1] High-pass filter: cutoff = {highpass_freq_min} Hz")

    # ------------------------------------------------------------------
    # Step 2: Inter-sample phase shift correction
    # NP2.0 channels are not sampled simultaneously — the ASIC reads groups
    # of channels in sequence within each 1/30kHz window, introducing
    # sub-sample timing offsets.  Phase shift corrects this before any
    # spatial operation (CAR, destripe) that compares across channels.
    # ------------------------------------------------------------------
    # rec = si.phase_shift(rec)
    # if verbose:
    print("[2] Phase shift correction is skipped for now (spikeinterface.preprocessing.phase_shift is not yet available)")

    # ------------------------------------------------------------------
    # Step 3: Bad channel detection and spatial interpolation
    # ------------------------------------------------------------------
    bad_channel_ids, channel_labels = si.detect_bad_channels(
        rec,
        # method="std",
        # std_mad_threshold=bad_channel_score_threshold,
    )
    if verbose:
        label_counts = {
            label: list(channel_labels).count(label)
            for label in set(channel_labels)
        }
        print(f"[3] Bad channels detected: {len(bad_channel_ids)}")
        print(f"    Label counts: {label_counts}")
        if len(bad_channel_ids):
            print(f"    Bad channel IDs: {list(bad_channel_ids)}")

    if len(bad_channel_ids) > 0:
        rec = si.interpolate_bad_channels(rec, bad_channel_ids)
        if verbose:
            print(f"    Interpolated {len(bad_channel_ids)} bad channels "
                  f"from spatial neighbours")
    else:
        if verbose:
            print("    No bad channels — skipping interpolation")

    # ------------------------------------------------------------------
    # Step 4: Spatial artifact removal
    # ------------------------------------------------------------------
    if artifact_removal == "car_global":
        rec = si.common_reference(rec, operator="median", reference="global")
        if verbose:
            print("[4] Spatial artefact removal: global CAR (median)")

    elif artifact_removal == "car_local":
        rec = si.common_reference(
            rec,
            operator="median",
            reference="local",
            local_radius=(car_local_inner_radius_um, car_local_outer_radius_um),
        )
        if verbose:
            print(f"[4] Spatial artefact removal: local CAR "
                  f"(annulus {car_local_inner_radius_um}–"
                  f"{car_local_outer_radius_um} µm)")

    elif artifact_removal == "destripe":
        rec = si.highpass_spatial_filter(rec)
        if verbose:
            print("[4] Spatial artefact removal: IBL destripe "
                  "(highpass_spatial_filter)")

    else:
        raise ValueError(
            f"Unknown artifact_removal strategy: '{artifact_removal}'. "
            "Choose 'car_global', 'car_local', or 'destripe'."
        )

    if verbose:
        print("\nPreprocessing chain built (lazy). Call .save() or pass to "
              "run_sorter() to execute.")

    return rec, list(bad_channel_ids)

def check_drift(
    recording: BaseRecording,
    n_jobs: int = 4,
    chunk_duration: str = "1s",
    detect_threshold: float = 5.0,
    radius_um: float = 50.0,
) -> tuple:
    """
    Estimate probe drift from peak locations before spike sorting.
    Produces a scatter plot (depth vs time) to visually inspect motion.

    Returns
    -------
    peaks       : structured ndarray of detected peaks
    locations   : ndarray of peak (x, y) positions in µm
    """
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration=chunk_duration,
                      progress_bar=True)

    noise_levels = si.get_noise_levels(recording, return_in_uV=False)

    print("Detecting peaks for drift estimation...")
    peaks = detect_peaks(
        recording,
        method="locally_exclusive",
        noise_levels=noise_levels,
        detect_threshold=detect_threshold,
        radius_um=radius_um,
        **job_kwargs,
    )
    print(f"  Detected {len(peaks):,} peaks")

    print("Localising peak positions...")
    locations = localize_peaks(
        recording,
        peaks,
        method="center_of_mass",
        radius_um=radius_um,
        **job_kwargs,
    )

    return peaks, locations