#!/usr/bin/env python3
"""
captions_to_firstframes_from_generated_videos.py

Input JSON (list of dicts), e.g.:
[
  {
    "caption": "A dog running through a field while a bird flying across the sky.",
    "phrase_0": [...],
    "phrase_1": [...]
  },
  ...
]

Assumes each caption produced a video at:
  <videos_root>/<Caption with spaces->underscores>.mp4
Example:
  samples/wan-videos-vbench-compbench/T2V/BASE/action_binding/
    A_dog_running_through_a_field_while_a_bird_flying_across_the_sky..mp4

This script:
- Locates the video for each caption (robust filename variants + glob fallbacks).
- Extracts the FIRST frame and saves it to --images_dir using the SAME basename as the video.
- Writes an output JSON list with items: {"file_name": "<image filename>", "caption": "<original caption>"}.
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import List, Optional

import imageio.v3 as iio
from PIL import Image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def caption_to_basename_variants(caption: str) -> List[str]:
    """
    Return plausible basenames (no extension) derived from the caption.
    Prioritize 'spaces->underscores, keep punctuation' to match examples,
    then add progressively sanitized fallbacks.
    """
    s = caption.strip()

    # Variant 1: spaces -> underscores, keep punctuation exactly
    v1 = s.replace(" ", "_")

    # Variant 2: collapse multiple underscores
    v2 = re.sub(r"_+", "_", v1)

    # Variant 3: strip all except letters, numbers, underscore, dot, dash
    v3 = re.sub(r"[^A-Za-z0-9._-]", "", v2)

    # Variant 4: remove trailing dots
    v4 = v3.rstrip(".")

    # Variant 5: lowercase (some setups downcase)
    v5 = v3.lower()

    seen, out = set(), []
    for v in [v1, v2, v3, v4, v5]:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def find_video_for_caption(
    caption: str, videos_root: str, ext: str = ".mp4"
) -> Optional[str]:
    """
    Try several filename variants to locate the video in videos_root.
    If direct checks fail, try a glob prefix match as a last resort.
    """
    variants = caption_to_basename_variants(caption)

    # Direct checks
    for base in variants:
        candidate = os.path.join(videos_root, base + ext)
        if os.path.isfile(candidate):
            return candidate

    # Glob fallback (prefix match)
    for base in variants:
        pattern = os.path.join(videos_root, base + "*" + ext)
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]

    # Very loose glob by first few words
    words = re.findall(r"[A-Za-z0-9]+", caption)
    if words:
        prefix = "_".join(words[:4])
        pattern = os.path.join(videos_root, f"{prefix}*{ext}")
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]

    return None


def save_first_frame(video_path: str, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Try index=0
    try:
        frame = iio.imread(video_path, index=0)
    except Exception:
        # Fallback to iterator
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

    try:
        img.save(out_path)
    except ValueError:
        fmt = os.path.splitext(out_path)[1][1:].upper() or "PNG"
        img.save(out_path, format=fmt)
    except Exception as e:
        eprint(f"[error] Failed saving image: {out_path} ({e})")
        return False

    # Verify saved image
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
        description="Extract first frames from generated videos and output JSON with file_name and caption."
    )
    ap.add_argument(
        "--in_json", required=True, help="Input JSON with 'caption' fields."
    )
    ap.add_argument(
        "--videos_root",
        required=True,
        help="Directory containing the generated videos.",
    )
    ap.add_argument(
        "--images_dir", required=True, help="Directory to save first-frame images."
    )
    ap.add_argument("--out_json", required=True, help="Output JSON path.")
    ap.add_argument(
        "--vid_ext",
        default=".mp4",
        choices=[".mp4", ".webm", ".mkv", ".mov"],
        help="Expected video extension.",
    )
    ap.add_argument(
        "--ext",
        default=".png",
        choices=[".png", ".jpg", ".jpeg"],
        help="Image extension for outputs.",
    )
    ap.add_argument(
        "--skip_existing", action="store_true", help="Skip if image already exists."
    )
    ap.add_argument(
        "--strict", action="store_true", help="Abort on first error if set."
    )
    args = ap.parse_args()

    # Load input
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

    out_list = []
    total = len(items)
    saved = skipped = failed = 0

    for idx, obj in enumerate(items, 1):
        caption = obj.get("caption", "")
        if not caption:
            msg = "[warn] Missing 'caption'; skipping."
            if args.strict:
                eprint(msg)
                sys.exit(1)
            eprint(msg)
            failed += 1
            continue

        video_path = find_video_for_caption(caption, args.videos_root, ext=args.vid_ext)
        if not video_path:
            msg = f"[warn] Could not find video for caption: {caption}"
            if args.strict:
                eprint(msg)
                sys.exit(1)
            eprint(msg)
            failed += 1
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        image_filename = base + args.ext.lower()
        image_out_path = os.path.join(args.images_dir, image_filename)

        if args.skip_existing and os.path.isfile(image_out_path):
            out_list.append({"file_name": image_filename, "caption": caption})
            skipped += 1
        else:
            ok = save_first_frame(video_path, image_out_path)
            if not ok:
                if args.strict:
                    sys.exit(1)
                failed += 1
                continue
            out_list.append({"file_name": image_filename, "caption": caption})
            saved += 1

        if idx % 50 == 0 or idx == total:
            eprint(
                f"[info] {idx}/{total} (saved={saved}, skipped={skipped}, failed={failed})"
            )

    # Write output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)

    eprint(
        f"[done] Wrote {len(out_list)} items to {args.out_json} (saved={saved}, skipped={skipped}, failed={failed})"
    )


if __name__ == "__main__":
    main()
