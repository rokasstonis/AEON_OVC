#!/bin/bash -l
# =============================================================
# submit_sleap_concat.sh — runs after the array job completes
#
# Do NOT submit this directly. It is submitted automatically
# by submit_sleap.sh with a --dependency=afterok on the array.
# =============================================================

#SBATCH --job-name=sleap_concat
#SBATCH --partition=gpu_branco
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ceph/branco/Rokas/Ephys/Results/Behaviour/logs/sleap_concat_%j.out
#SBATCH --error=/ceph/branco/Rokas/Ephys/Results/Behaviour/logs/sleap_concat_%j.err

VIDEO_LIST="${1}"
RESULTS_DIR="${2}"
SESSION_ID="${3}"
OUTPUT_FILE="${RESULTS_DIR}/${SESSION_ID}_predictions.h5"

echo "============================================"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Started   : $(date)"
echo "Session   : ${SESSION_ID}"
echo "Output    : ${OUTPUT_FILE}"
echo "============================================"

module load SLEAP/2023-08-01

python - << EOF
import h5py
import numpy as np
from pathlib import Path

video_list  = Path("${VIDEO_LIST}")
results_dir = Path("${RESULTS_DIR}")
output_file = Path("${OUTPUT_FILE}")
session_id  = "${SESSION_ID}"

# Read video list and find corresponding h5 files, preserving order
videos = [Path(l.strip()) for l in video_list.read_text().splitlines() if l.strip()]
h5_files = []
missing  = []

for v in videos:
    h5 = results_dir / f"{v.stem}.h5"
    if h5.exists() and h5.stat().st_size > 0:
        h5_files.append(h5)
    else:
        missing.append(v.stem)

if missing:
    print(f"ERROR: {len(missing)} videos have no predictions file:")
    for m in missing:
        print(f"  {m}")
    print("Re-run the array job for these videos before concatenating.")
    exit(1)

print(f"Concatenating {len(h5_files)} prediction files in temporal order...")

# -------------------------------------------------------------------
# Read structure from first file to understand the layout
# SLEAP h5 analysis files contain:
#   /tracks          (n_frames, n_nodes, 2, n_tracks)  — x,y per node per animal
#   /track_names     list of track labels
#   /node_names      list of skeleton node names
#   /point_scores    (n_frames, n_nodes, n_tracks)     — confidence per point
#   /instance_scores (n_frames, n_tracks)              — instance confidence
#   /tracking_scores (n_frames, n_tracks)
#   /video_path      string
# -------------------------------------------------------------------

with h5py.File(h5_files[0], "r") as f:
    node_names  = [n.decode() if isinstance(n, bytes) else n
                   for n in f["node_names"][:]]
    track_names = [n.decode() if isinstance(n, bytes) else n
                   for n in f["track_names"][:]]
    n_nodes  = len(node_names)
    n_tracks = len(track_names)
    print(f"  Nodes  : {node_names}")
    print(f"  Tracks : {track_names}")

# Count total frames
total_frames = 0
frame_counts = []
for h5 in h5_files:
    with h5py.File(h5, "r") as f:
        n = f["tracks"].shape[0]
        frame_counts.append(n)
        total_frames += n

print(f"  Total frames: {total_frames:,}  across {len(h5_files)} files")

# Pre-allocate output arrays
tracks          = np.full((total_frames, n_nodes, 2, n_tracks), np.nan, dtype=np.float32)
point_scores    = np.full((total_frames, n_nodes, n_tracks),    np.nan, dtype=np.float32)
instance_scores = np.full((total_frames, n_tracks),             np.nan, dtype=np.float32)
tracking_scores = np.full((total_frames, n_tracks),             np.nan, dtype=np.float32)

# Fill chunk by chunk
offset = 0
for h5, n_frames in zip(h5_files, frame_counts):
    with h5py.File(h5, "r") as f:
        tracks[offset:offset+n_frames]          = f["tracks"][:]
        point_scores[offset:offset+n_frames]    = f["point_scores"][:]
        instance_scores[offset:offset+n_frames] = f["instance_scores"][:]
        if "tracking_scores" in f:
            tracking_scores[offset:offset+n_frames] = f["tracking_scores"][:]
    offset += n_frames
    print(f"  Loaded {h5.name} ({n_frames:,} frames)")

# -------------------------------------------------------------------
# Write concatenated output
# -------------------------------------------------------------------
print(f"\nWriting {output_file} ...")
with h5py.File(output_file, "w") as f:
    f.create_dataset("tracks",          data=tracks,          compression="gzip")
    f.create_dataset("point_scores",    data=point_scores,    compression="gzip")
    f.create_dataset("instance_scores", data=instance_scores, compression="gzip")
    f.create_dataset("tracking_scores", data=tracking_scores, compression="gzip")

    # Store metadata
    f.create_dataset("node_names",  data=np.array(node_names,  dtype="S"))
    f.create_dataset("track_names", data=np.array(track_names, dtype="S"))

    # Frame offsets — useful for mapping back to individual video files
    f.create_dataset("frame_offsets", data=np.cumsum([0] + frame_counts[:-1]))
    f.create_dataset("frame_counts",  data=np.array(frame_counts))

    # Source video paths in order
    video_paths = [str(results_dir / f"{v.stem}.h5") for v in videos]
    f.create_dataset("source_files", data=np.array(video_paths, dtype="S"))

    # Sampling rate placeholder (fill in if known)
    f.attrs["session_id"]     = session_id
    f.attrs["n_frames"]       = total_frames
    f.attrs["n_nodes"]        = n_nodes
    f.attrs["n_tracks"]       = n_tracks
    f.attrs["n_source_files"] = len(h5_files)

size_gb = output_file.stat().st_size / 1e9
print(f"Done. Output: {output_file}  ({size_gb:.2f} GB)")
print(f"\nTo load in Python:")
print(f"  import h5py, numpy as np")
print(f"  with h5py.File('{output_file}', 'r') as f:")
print(f"      tracks = f['tracks'][:]  # shape: (n_frames, n_nodes, 2, n_tracks)")
print(f"      scores = f['point_scores'][:]")
EOF

EXIT_CODE=$?
echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
