#!/usr/bin/env python3
"""
scripts/hpc/check_sleap_progress.py
------------------------------------
Check which videos completed, which failed, and write a new
video_list.txt containing only the failures for resubmission.

Usage
-----
    python scripts/hpc/check_sleap_progress.py \
        --video-list /ceph/branco/Rokas/Ephys/Results/video_list.txt \
        --results-dir /ceph/branco/Rokas/Ephys/Results/sleap_predictions \
        --output-format h5
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-list",    required=True)
    p.add_argument("--results-dir",   required=True)
    p.add_argument("--output-format", default="h5", choices=["h5", "slp"])
    args = p.parse_args()

    video_list  = Path(args.video_list)
    results_dir = Path(args.results_dir)
    ext         = args.output_format

    videos = [Path(l.strip()) for l in video_list.read_text().splitlines() if l.strip()]

    completed, failed, missing_input = [], [], []

    for v in videos:
        stem = v.stem
        out  = results_dir / f"{stem}.{ext}"
        if not v.exists():
            missing_input.append(v)
        elif out.exists() and out.stat().st_size > 0:
            completed.append(v)
        else:
            failed.append(v)

    print(f"Total videos    : {len(videos)}")
    print(f"Completed       : {len(completed)}")
    print(f"Failed/missing  : {len(failed)}")
    print(f"Input not found : {len(missing_input)}")

    if failed:
        retry_path = video_list.parent / "video_list_retry.txt"
        retry_path.write_text("\n".join(str(f) for f in failed) + "\n")
        print(f"\nRetry list written to: {retry_path}")
        print(f"Update VIDEO_LIST in submit_sleap.sh to point at this file,")
        print(f"and set --array=1-{len(failed)}%20, then resubmit.")

    if missing_input:
        print("\nVideos with missing input files:")
        for v in missing_input:
            print(f"  {v}")


if __name__ == "__main__":
    main()
