#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def load_upsampled_map(upsampled_root: Path):
    """
    Build {dimension: [ {caption, original_caption}, ... ]} from
    files like: <upsampled_root>/<Dimension>_longer.json
    """
    up_map = {}
    for p in sorted(upsampled_root.glob("*_longer.json")):
        dim = p.stem.replace("_longer", "")
        with p.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Upsampled file is not a list: {p}")
        up_map[dim] = payload
    return up_map


def process_json_file(in_path: Path, out_path: Path, up_entries: list):
    """
    Replace each dict's 'caption' with the upsampled caption:
      - original_caption := previous caption (unless already present)
      - caption := upsampled_entries[entry_idx or position]['caption']
    """
    with in_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        # Some files might be a dict; handle conservatively by skipping.
        # Adjust as needed if you actually have dict-shaped files to update.
        return False, f"Skipped (not a list JSON): {in_path}"

    new_data = []
    for pos, item in enumerate(data):
        if not isinstance(item, dict):
            new_data.append(item)
            continue

        # Prefer entry_idx when available; else use position
        idx = item.get("entry_idx", pos)

        if idx < 0 or idx >= len(up_entries):
            # If out-of-range, leave it as-is but add note
            new_data.append(item)
            continue

        up = up_entries[idx]
        # Keep the original caption under 'original_caption'
        if "caption" in item and "original_caption" not in item:
            item["original_caption"] = item["caption"]

        # Overwrite caption with upsampled one
        item["caption"] = up.get("caption", item.get("caption", ""))

        new_data.append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return True, None


def main():
    ap = argparse.ArgumentParser(
        description="Replace captions with upsampled prompts across a mirrored folder tree."
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root of the original JSON tree (e.g., VBench2_augmented_json)",
    )
    ap.add_argument(
        "--upsampled",
        required=True,
        type=Path,
        help="Folder with *_longer.json files (e.g., VBench-2.0/prompts/prompts_per_dimension_json_longer_with_og)",
    )
    ap.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Output root to write the mirrored tree with updated captions",
    )
    ap.add_argument(
        "--ext", default=".json", help="File extension to process (default: .json)"
    )
    args = ap.parse_args()

    up_map = load_upsampled_map(args.upsampled)

    # Walk: <src>/<Dimension>/<subfolder>/**/*.json
    for dim_dir in sorted(args.src.iterdir()):
        if not dim_dir.is_dir():
            continue
        dimension = dim_dir.name  # e.g., "Camera_Motion"
        up_entries = up_map.get(dimension)

        if up_entries is None:
            print(
                f"[WARN] No upsampled file found for dimension '{dimension}'. Skipping this dimension."
            )
            continue

        for json_path in dim_dir.rglob(f"*{args.ext}"):
            # Mirror path from src to dst
            rel = json_path.relative_to(args.src)
            out_path = args.dst / rel

            ok, err = process_json_file(json_path, out_path, up_entries)
            if not ok:
                print(f"[INFO] {err}")


if __name__ == "__main__":
    main()
