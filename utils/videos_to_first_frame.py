#!/usr/bin/env python3
"""
videos_to_firstframes_json.py

Input JSON format (list of dicts):
[
  {"video_path": "/path/to/video.mp4", "caption": "\"Some caption ...\",6.03,2.04,0.99,static,97,25.0,3.88"},
  ...
]

This script:
- Extracts the first frame of each video to an images directory, using the same base name as the video (e.g., foo.mp4 -> foo.png).
- Cleans the caption so only the textual caption remains.
- Writes an output JSON: [{"image_path": "...", "caption": "..."}, ...]

Based on your working frame-extraction approach (imageio + PIL).
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import Dict, List

import imageio.v3 as iio
from PIL import Image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def clean_caption(raw: str) -> str:
    """
    Your input caption is often a CSV-like string inside JSON, e.g.:
      "\"The video ... Mustang.\",6.1144,3.5886,0.99,zoom_out,130,23.976,5.42"
    We parse it as a CSV row and return the first field.
    Falls back to simple split if CSV parsing fails.
    """
    if not isinstance(raw, str):
        return str(raw)

    s = raw.strip()

    # Try robust CSV parsing (handles doubled quotes inside the quoted field)
    try:
        row = next(csv.reader(io.StringIO(s)))
        if row:
            return row[0].strip()
    except Exception:
        pass

    # Fallback: split at '",' and strip surrounding quotes
    if '",' in s:
        head = s.split('",', 1)[0]
        return head.strip().strip('"')

    # Final fallback: return as-is
    return s


def ensure_image_path(video_path: str, images_dir: str, ext: str = ".png") -> str:
    """
    Build the output image path: <images_dir>/<video_stem>.png
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(images_dir, base + ext.lower())
    return out


def save_first_frame(video_path: str, out_path: str) -> bool:
    """
    Extract first frame using imageio and save via PIL.
    Returns True on success, False on failure.
    """
    # Ensure parent dir
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Grab first frame
    try:
        frame = iio.imread(video_path, index=0)
    except Exception:
        try:
            it = iio.imiter(video_path)
            frame = next(it)
        except Exception as e:
            eprint(f"[error] Failed reading first frame: {video_path} ({e})")
            return False

    try:
        img = Image.fromarray(frame)
    except Exception as e:
        eprint(f"[error] Not an image array for: {video_path} ({e})")
        return False

    # Save; if PIL complains about extension, pass format explicitly
    try:
        img.save(out_path)
    except ValueError:
        fmt = os.path.splitext(out_path)[1][1:].upper() or "PNG"
        img.save(out_path, format=fmt)
    except Exception as e:
        eprint(f"[error] Failed saving image: {out_path} ({e})")
        return False

    # Quick sanity check: reopen to ensure it’s valid
    try:
        with Image.open(out_path) as check_img:
            check_img.verify()
    except Exception as e:
        eprint(f"[error] Saved image seems invalid: {out_path} ({e})")
        try:
            os.remove(out_path)
        except OSError:
            pass
        return False

    return True


def main():
    ap = argparse.ArgumentParser(
        description="Extract first frames and make image+caption JSON."
    )
    ap.add_argument(
        "--in_json",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/OpenVid-1M-val.json",
        help="Path to input JSON with video_path + caption.",
    )
    ap.add_argument(
        "--images_dir",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/validation_images",
        help="Directory to save extracted first frames.",
    )
    ap.add_argument(
        "--out_json",
        default="/capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/OpenVid-1M-val-imgs.json",
        help="Output JSON path with image_path + caption.",
    )
    ap.add_argument(
        "--ext",
        default=".png",
        choices=[".png", ".jpg", ".jpeg"],
        help="Image extension for outputs.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip extraction if target image already exists.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, abort on any error; otherwise skip bad items.",
    )
    args = ap.parse_args()

    # Load input list
    if not os.path.isfile(args.in_json):
        eprint(f"[error] Input JSON not found: {args.in_json}")
        sys.exit(1)

    try:
        with open(args.in_json, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        eprint(f"[error] Failed to parse JSON: {e}")
        sys.exit(1)

    if not isinstance(items, list):
        eprint("[error] Input JSON must be a list of objects.")
        sys.exit(1)

    os.makedirs(args.images_dir, exist_ok=True)

    out_items: List[Dict[str, str]] = []
    total = len(items)
    done = 0
    skipped = 0
    failures = 0

    for idx, obj in enumerate(items, 1):
        video_path = obj.get("video_path", "")
        raw_caption = obj.get("caption", "")

        if not video_path or not os.path.isfile(video_path):
            msg = f"[warn] Missing video file: {video_path}"
            if args.strict:
                eprint(msg)
                sys.exit(1)
            eprint(msg + " (skipping)")
            failures += 1
            continue

        image_path = ensure_image_path(video_path, args.images_dir, ext=args.ext)

        if args.skip_existing and os.path.isfile(image_path):
            cleaned = clean_caption(raw_caption)
            out_items.append(
                {"image_path": os.path.abspath(image_path), "caption": cleaned}
            )
            skipped += 1
            continue

        ok = save_first_frame(video_path, image_path)
        if not ok:
            if args.strict:
                sys.exit(1)
            failures += 1
            continue

        cleaned = clean_caption(raw_caption)
        out_items.append(
            {"image_path": os.path.abspath(image_path), "caption": cleaned}
        )
        done += 1

        if idx % 50 == 0 or idx == total:
            eprint(
                f"[info] Progress: {idx}/{total} (saved={done}, skipped={skipped}, failed={failures})"
            )

    # Write output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    eprint(
        f"[done] Wrote {len(out_items)} items to {args.out_json} "
        f"(saved={done}, skipped={skipped}, failed={failures})"
    )


if __name__ == "__main__":
    main()
