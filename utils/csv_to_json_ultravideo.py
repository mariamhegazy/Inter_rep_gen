#!/usr/bin/env python3
"""
csv_to_training_json.py

Read a CSV file containing UltraVideo metadata and produce
a JSON training file containing entries of the form:
{"file_path": "<clip_id>", "text": "<Brief Description>", "type": "video", "id": "<clip_id>"}

Only include entries whose video files actually exist in the provided directory.
"""

import argparse
import csv
import json
import os
import sys

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def index_videos(videos_dir: str, recursive: bool = True):
    """Return a set of all available video filenames."""
    found = set()
    walker = (
        os.walk(videos_dir) if recursive else [(videos_dir, [], os.listdir(videos_dir))]
    )
    for root, _, files in walker:
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                found.add(f)
    return found


def main():
    ap = argparse.ArgumentParser(description="Convert UltraVideo CSV to training JSON.")
    ap.add_argument(
        "--csv",
        default="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/short.csv",
        help="Path to the input CSV file.",
    )
    ap.add_argument(
        "--videos_dir",
        default="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/clips_short/clips_short",
        help="Directory containing the video files.",
    )
    ap.add_argument(
        "--out",
        default="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training.json",
        help="Output JSON file path.",
    )
    ap.add_argument(
        "--no_recursive", action="store_true", help="Disable recursive search."
    )
    args = ap.parse_args()

    # Validate input paths
    if not os.path.isfile(args.csv):
        eprint(f"[error] CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.videos_dir):
        eprint(f"[error] videos_dir not found: {args.videos_dir}")
        sys.exit(1)

    # Index available videos
    video_names = index_videos(args.videos_dir, recursive=not args.no_recursive)
    eprint(f"[info] Found {len(video_names)} video files in {args.videos_dir}")

    # Read the CSV
    items = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row.get("clip_id")
            # desc = row.get("Brief Description")
            desc = row.get("Summarized Description")
            if not clip_id or not desc:
                continue
            if clip_id not in video_names:
                continue  # skip missing videos

            items.append(
                {
                    "file_path": clip_id,
                    "text": desc.strip(),
                    "type": "video",
                    "id": os.path.splitext(clip_id)[0],
                }
            )

    # Save output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    eprint(f"[done] Wrote {len(items)} valid items to {args.out}")


if __name__ == "__main__":
    main()
