#!/usr/bin/env python3
"""
csv_to_video_caption_json.py

Read a 2-column CSV/TSV (filename, caption), look up each video inside a folder,
and write out a JSON list of {"video_path": "...", "caption": "..."}.

Robust to:
- Tabs or commas (splits on the *first* delimiter only)
- Leading dashes/underscores accidentally prefixed to filenames (e.g., "---foo.mp4")
- Recursive directories of videos
- UTF-8 (with BOM) and common encodings
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sanitize_name(name: str) -> str:
    """Trim whitespace/quotes/BOM and strip leading -/_ that may appear in the CSV."""
    s = name.strip().strip('"').strip("'").replace("\ufeff", "")
    # Some rows may start with many '-' or '_' (e.g., '---video.mp4'); strip them.
    s = s.lstrip("-_")
    return s


def detect_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """Read file as lines, trying utf-8 first and falling back to latin-1."""
    encodings = [encoding, "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read().splitlines()
        except Exception as e:
            last_err = e
    raise last_err


def parse_two_columns(lines: List[str]) -> List[Tuple[str, str]]:
    """
    Parse a list of lines into (filename, caption).
    Splits on first '\t', else on first ',', else on first whitespace.
    Ignores empty/comment lines.
    """
    pairs = []
    for idx, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        # Allow comment lines if present
        if line.startswith("#"):
            continue

        left = right = None
        if "\t" in line:
            left, right = line.split("\t", 1)
        elif "," in line:
            left, right = line.split(",", 1)
        else:
            parts = line.split(None, 1)  # split on first whitespace block
            if len(parts) == 2:
                left, right = parts
        if left is None or right is None:
            eprint(f"[warn] Could not parse line {idx}: {raw[:120]}...")
            continue

        left = sanitize_name(left)
        right = right.strip()
        pairs.append((left, right))
    return pairs


def index_videos(videos_dir: str, recursive: bool = True) -> Dict[str, str]:
    """
    Walk the directory and create lookup maps by:
      - exact basename (lowercase)
      - sanitized basename (leading -/_ removed)
    Returns a dict from keys to full absolute paths. If duplicates occur, first is kept.
    """
    index: Dict[str, str] = {}

    def add_key(key: str, path: str):
        key_l = key.lower()
        if key_l not in index:
            index[key_l] = path

    walker = (
        os.walk(videos_dir) if recursive else [(videos_dir, [], os.listdir(videos_dir))]
    )
    for root, _, files in walker:
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            full = os.path.abspath(os.path.join(root, f))
            base = os.path.basename(full)
            add_key(base, full)
            add_key(sanitize_name(base), full)
    return index


def resolve_video_path(name: str, index: Dict[str, str], videos_dir: str) -> str:
    """
    Given a (possibly messy) name from CSV and an index of videos,
    try to resolve to a full path.
    """
    candidates = [
        name,
        sanitize_name(name),
        os.path.basename(name),
        os.path.basename(sanitize_name(name)),
    ]
    for c in candidates:
        key = c.lower()
        if key in index:
            return index[key]

    # As a last resort, try direct join (if user gave a relative that exists)
    direct = os.path.abspath(os.path.join(videos_dir, name))
    if os.path.exists(direct):
        return direct

    return ""  # not found


def main():
    ap = argparse.ArgumentParser(
        description="Make JSON of video paths and captions from a CSV/TSV."
    )
    ap.add_argument(
        "--csv",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/OpenVid-1M.csv",
        help="Path to the CSV/TSV file (2 columns: filename, caption).",
    )
    ap.add_argument(
        "--videos_dir",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/validation",
        help="Folder containing validation videos (searched recursively).",
    )
    ap.add_argument(
        "--out",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/OpenVid-1M-val.json",
        help="Output JSON path.",
    )
    ap.add_argument(
        "--no_recursive", action="store_true", help="Do not recurse into subfolders."
    )
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip entries whose video is not found.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        eprint(f"[error] CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.videos_dir):
        eprint(f"[error] videos_dir not found: {args.videos_dir}")
        sys.exit(1)

    lines = detect_lines(args.csv)
    pairs = parse_two_columns(lines)
    if not pairs:
        eprint("[error] No (filename, caption) pairs parsed from CSV.")
        sys.exit(1)

    eprint(f"[info] Parsed {len(pairs)} rows from CSV.")
    index = index_videos(args.videos_dir, recursive=not args.no_recursive)
    eprint(f"[info] Indexed {len(index)} video keys under: {args.videos_dir}")

    result = []
    missing = 0
    for fname, caption in pairs:
        path = resolve_video_path(fname, index, args.videos_dir)
        if not path:
            missing += 1
            msg = f"[warn] Missing video for '{fname}'"
            if args.skip_missing:
                eprint(msg + " (skipping)")
                continue
            else:
                eprint(msg + " (keeping with empty path)")
        item = {"video_path": path, "caption": caption}
        result.append(item)

    # Sort for determinism (by basename)
    result.sort(key=lambda x: os.path.basename(x["video_path"] or ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    eprint(f"[done] Wrote {len(result)} items to {args.out}. Missing videos: {missing}")


if __name__ == "__main__":
    main()
