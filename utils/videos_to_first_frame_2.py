#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import imageio.v3 as iio
from PIL import Image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def resolve_video_path(vid: str, videos_dir: str, try_exts: List[str]) -> Optional[str]:
    # If user gave an absolute/relative path and it exists
    if os.path.isabs(vid) and os.path.isfile(vid):
        return vid

    # Try the name as given under videos_dir (handles when vid already has .mp4)
    as_given = os.path.join(videos_dir, vid)
    if os.path.isfile(as_given):
        return as_given

    # If vid has an extension, try with the same basename + the allowed exts
    base, ext = os.path.splitext(vid)
    bases = [base] if ext else [vid]

    for b in bases:
        for e in try_exts:
            cand = os.path.join(videos_dir, b + e)
            if os.path.isfile(cand):
                return cand

    return None


def ensure_image_path(vid: str, images_dir: str, ext: str = ".png") -> str:
    return os.path.join(images_dir, vid + ext.lower())


def save_first_frame(video_path: str, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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
    try:
        img.save(out_path)
    except ValueError:
        fmt = os.path.splitext(out_path)[1][1:].upper() or "PNG"
        img.save(out_path, format=fmt)
    except Exception as e:
        eprint(f"[error] Failed saving image: {out_path} ({e})")
        return False
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
        description="Extract first frames and make image-filename + caption JSON."
    )
    ap.add_argument(
        "--in_json", required=True, help="Input JSON with [{vid, caption}, ...]."
    )
    ap.add_argument(
        "--videos_dir",
        required=True,
        help="Directory containing videos named as <vid>.<ext>.",
    )
    ap.add_argument(
        "--images_dir", required=True, help="Directory to save extracted first frames."
    )
    ap.add_argument(
        "--out_json",
        required=True,
        help="Output JSON path [{image_filename, caption}, ...].",
    )
    ap.add_argument(
        "--img_ext",
        default=".png",
        choices=[".png", ".jpg", ".jpeg"],
        help="Image extension for outputs.",
    )
    ap.add_argument(
        "--video_exts",
        default=".mp4,.webm,.mkv,.mov",
        help="Comma-separated list of video extensions to try, in order.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip extraction if target image already exists.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Abort on any error; otherwise skip bad items.",
    )
    args = ap.parse_args()

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
    try_exts = [
        e if e.startswith(".") else "." + e
        for e in args.video_exts.split(",")
        if e.strip()
    ]

    out_items: List[Dict[str, str]] = []
    total = len(items)
    done = skipped = failures = 0

    for idx, obj in enumerate(items, 1):
        vid = obj.get("vid", "")
        caption = obj.get("caption", "")

        if not vid:
            msg = "[warn] Missing 'vid' field in item"
            if args.strict:
                eprint(msg)
                sys.exit(1)
            eprint(msg + " (skipping)")
            failures += 1
            continue

        video_path = resolve_video_path(vid, args.videos_dir, try_exts)
        if not video_path:
            msg = f"[warn] Video not found for vid={vid} in {args.videos_dir} with exts {try_exts}"
            if args.strict:
                eprint(msg)
                sys.exit(1)
            eprint(msg + " (skipping)")
            failures += 1
            continue

        image_path = ensure_image_path(vid, args.images_dir, ext=args.img_ext)
        image_filename = os.path.basename(image_path)

        if args.skip_existing and os.path.isfile(image_path):
            out_items.append({"file_name": image_filename, "caption": str(caption)})
            skipped += 1
        else:
            ok = save_first_frame(video_path, image_path)
            if not ok:
                if args.strict:
                    sys.exit(1)
                failures += 1
                continue
            out_items.append({"file_name": image_filename, "caption": str(caption)})
            done += 1

        if idx % 50 == 0 or idx == total:
            eprint(
                f"[info] Progress: {idx}/{total} (saved={done}, skipped={skipped}, failed={failures})"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    eprint(
        f"[done] Wrote {len(out_items)} items to {args.out_json} (saved={done}, skipped={skipped}, failed={failures})"
    )


if __name__ == "__main__":
    main()
