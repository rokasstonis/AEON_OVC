"""Run foragingABC's SLEAP pipeline with legacy SLEAP 1.4 (sleap-track) inference.

foragingabc.dj_pipeline.pose.SleapRawTracking calls
utils.sleap.pipeline.run_sleap_pipeline, which runs `sleap track` (sleap-nn,
SLEAP >=1.5) via the module-level name `run_inference`. Our models perform
poorly under sleap-nn, so apply() swaps that name for run_inference_legacy,
which has the same signature and runs `sleap-track` from the sleap_legacy
conda env with the flags from scripts/hpc/submit_sleap.sh.

Everything else (pre-checks, .slp -> movement xarray, HARP alignment,
post-checks, keep_slp) is foragingABC's unchanged.

Usage (before any populate call):
    import sleap_legacy
    sleap_legacy.apply()
"""

import os
import shlex
import subprocess
import time
from pathlib import Path

CONDA_BASE = os.environ.get("SLEAP_LEGACY_CONDA_BASE", "/ceph/apps/ubuntu-20/packages/miniconda/23.10.0")
CONDA_ENV = os.environ.get("SLEAP_LEGACY_CONDA_ENV", "sleap_legacy")
DEFAULT_BATCH_SIZE = 8

# Flags from scripts/hpc/submit_sleap.sh (single animal)
TRACK_FLAGS = [
    "--max_tracks", "1",
    "--max_instances", "1",
    "--peak-threshold", "0.2",
    "--verbosity", "json",
]


def _legacy_env() -> dict[str, str]:
    """Environment for the legacy subprocess, without the main venv leaking in."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # let several processes share one GPU
    return env


def run_inference_legacy(
    video_path: Path,
    centroid_model: Path,
    centered_model: Path,
    output_slp: Path,
    log_path: Path,
    batch_size: int | None = None,
    threads_per_job: int | None = None,
    queue_maxsize: int = 128,  # sleap-nn only; accepted for signature compatibility
    silent_log: bool = False,
) -> None:
    """Drop-in replacement for foragingabc...sleap.track.run_inference."""
    from foragingabc.dj_pipeline.utils.sleap.utils import die, init_logger, print_info, print_ok, print_step

    init_logger(log_path, silent=silent_log)
    output_slp.parent.mkdir(parents=True, exist_ok=True)

    conda_sh = Path(CONDA_BASE) / "etc/profile.d/conda.sh"
    if not conda_sh.is_file():
        die(f"conda.sh not found at {conda_sh} (set SLEAP_LEGACY_CONDA_BASE)")

    if batch_size is None:
        batch_size = int(os.environ.get("SLEAP_BATCH_SIZE", DEFAULT_BATCH_SIZE))

    env = _legacy_env()
    if threads_per_job is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[var] = str(threads_per_job)

    # Write to a temp file and rename, so an interrupted run never leaves a
    # partial .slp that run_sleap_pipeline would later "reuse".
    tmp_slp = output_slp.with_name(output_slp.stem + ".partial.slp")
    track_cmd = [
        "sleap-track", str(video_path),
        "--model", str(centroid_model),
        "--model", str(centered_model),
        "--output", str(tmp_slp),
        "--batch-size", str(batch_size),
        *TRACK_FLAGS,
    ]
    # Activate the env the same way submit_sleap.sh does, so its CUDA libs are found
    shell_cmd = (
        f"source {shlex.quote(str(conda_sh))} && conda activate {shlex.quote(CONDA_ENV)} "
        f"&& echo \"sleap-track: $(command -v sleap-track)\" && {shlex.join(track_cmd)}"
    )

    print_step(f"Running legacy SLEAP inference  →  {video_path.name}")
    print_info(f"Conda env      : {CONDA_BASE} / {CONDA_ENV}")
    print_info(f"Centroid model : {centroid_model}")
    print_info(f"Centered model : {centered_model}")
    print_info(f"Output .slp    : {output_slp}")
    print_info(f"Batch size     : {batch_size}")

    t_start = time.time()
    with open(log_path, "a") as log:
        result = subprocess.run(["bash", "-c", shell_cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed_min = (time.time() - t_start) / 60

    if result.returncode != 0:
        tmp_slp.unlink(missing_ok=True)
        die(f"sleap-track exited with code {result.returncode}.\n     See log: {log_path}")
    if not tmp_slp.exists() or tmp_slp.stat().st_size == 0:
        die(f"sleap-track reported success but the .slp is missing or empty:\n     {tmp_slp}")

    tmp_slp.rename(output_slp)
    size_mb = output_slp.stat().st_size / 1024**2
    print_ok(f"Inference complete  ({size_mb:.1f} MB, {elapsed_min:.1f} min)  →  {output_slp.name}\n")


def apply() -> None:
    """Make SleapRawTracking use legacy inference. Idempotent."""
    from foragingabc.dj_pipeline.utils.sleap import pipeline

    pipeline.run_inference = run_inference_legacy
    print(f"sleap_legacy: inference via `sleap-track` from {CONDA_BASE}/envs/{CONDA_ENV}", flush=True)
