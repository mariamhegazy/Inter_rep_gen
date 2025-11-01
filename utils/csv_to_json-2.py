#!/usr/bin/env python3
"""
csv_to_name_caption_json.py

Read a CSV with video metadata and write a JSON list containing only:
  {"name": "...", "caption": "..."}

Auto-detects common column names:
  - name:  videoID | video_id | videoid | filename | name
           (fallback: extract YouTube ID from 'url'/'video_url')
  - caption: caption | text | description

Usage:
  python csv_to_name_caption_json.py --csv /path/to/file.csv --out out.json
  # (optional)
  python csv_to_name_caption_json.py --name_col videoID --caption_col caption --limit 10000
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

import pandas as pd


def find_col(cols_lower_map, *candidates) -> Optional[str]:
    """Return the original column name matching first candidate found (case-insensitive)."""
    for c in candidates:
        if c in cols_lower_map:
            return cols_lower_map[c]
    return None


def extract_youtube_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    # common patterns
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return None


def resolve_name(row, name_col, url_cols):
    if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip():
        return str(row[name_col]).strip()
    for uc in url_cols:
        if uc and uc in row and pd.notna(row[uc]):
            vid = extract_youtube_id(str(row[uc]))
            if vid:
                return vid
    # last resort: synthetic name
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Make JSON with only {name, caption} from a CSV."
    )
    ap.add_argument(
        "--csv",
        default="/capstor/store/cscs/swissai/a144/datasets/panda-70m/panda70m_training_2m.csv",
        help="Path to input CSV.",
    )
    ap.add_argument(
        "--out",
        default="/capstor/store/cscs/swissai/a144/datasets/panda-70m/videos.json",
        help="Path to output JSON.",
    )
    ap.add_argument(
        "--name_col",
        default=None,
        help="Explicit column for name (overrides auto-detect).",
    )
    ap.add_argument(
        "--caption_col",
        default=None,
        help="Explicit column for caption (overrides auto-detect).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of rows written.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[error] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Read CSV (pandas handles big files well; change chunksize if needed)
    df = pd.read_csv(args.csv)

    # Map lowercase->original for robust lookup
    cols_lower_map = {c.lower(): c for c in df.columns}

    # Resolve caption column
    caption_col = args.caption_col or find_col(
        cols_lower_map, "caption", "text", "description"
    )
    if not caption_col:
        print(
            f"[error] Could not find a caption column among {list(df.columns)}. "
            f"Use --caption_col to specify.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve name column or fall back to URL-derived YouTube ID
    name_col = args.name_col or find_col(
        cols_lower_map, "videoid", "video_id", "videoid", "filename", "name"
    )
    url_col = find_col(cols_lower_map, "url", "video_url")
    url_cols = [url_col] if url_col else []

    out = []
    count = 0
    for _, row in df.iterrows():
        name = resolve_name(row, name_col, url_cols)
        if not name:
            # Skip rows we cannot name (you can change to keep with None)
            continue
        cap = row[caption_col]
        if pd.isna(cap):
            cap = ""
        item = {"vid": str(name) + ".mp4", "caption": str(cap)}
        out.append(item)
        count += 1
        if args.limit and count >= args.limit:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        f"[done] Wrote {len(out)} items to {args.out} "
        f"(from {len(df)} rows; name_col={name_col or 'auto/url'}, caption_col={caption_col})."
    )


if __name__ == "__main__":
    main()
