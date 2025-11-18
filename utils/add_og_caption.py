#!/usr/bin/env python3
"""
Add `original_caption` to every JSON file under a metrics tree.

INPUT TREE (example):
json_metadata_upsampled/
  Camera_Motion/
    caption/
      0.json
      1.json
      ...
    first_frame/
      0.json
      1.json
      ...
  Complex_Landscape/
    ...

REFERENCE (per-metric, exactly one JSON each — either <root>/<metric>.json
or <root>/<metric>/<something>.json):
Vbench2_frames_captions_upsampled/
  Camera_Motion.json           # or
  Camera_Motion/
    frames.json                # (any *.json, first one found is used)
  ...

Each reference JSON is a list (or NDJSON) of objects like:
{
  "caption": "...",                # long full caption
  "original_caption": "Garden, zoom in.",
  "frame_first": "...",            # initial state text
  "frame_json_raw": "{...}"
}

We match per-file JSONs via `entry_idx` to the same row in the reference, then
copy `original_caption` to the file object and save to an output root, preserving
the folder structure.

Usage:
  python add_original_caption.py \
    --metadata-root /path/to/json_metadata_upsampled \
    --summary-root  /path/to/Vbench2_frames_captions_upsampled \
    --out-root      /path/to/json_metadata_upsampled_with_original

Options:
  --strict      : error if a metric has no summary (default: warn & skip)
  --verbose     : print detailed progress
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_summary_file(summary_root: Path, metric: str) -> Optional[Path]:
    # Try <root>/<metric>.json
    cand = summary_root / f"{metric}.json"
    if cand.is_file():
        return cand

    # Try <root>/<metric>/*.json (pick the first sorted)
    cand_dir = summary_root / metric
    if cand_dir.is_dir():
        jsons = sorted(cand_dir.glob("*.json"))
        if jsons:
            return jsons[0]

    # Fallback: any json in root that contains the metric name
    jsons = sorted(summary_root.glob(f"*{metric}*.json"))
    if jsons:
        return jsons[0]

    return None


def _load_json_loose(path: Path) -> Any:
    """Load JSON that could be a list, dict, or NDJSON lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try NDJSON (one JSON object per line)
        data = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            data.append(json.loads(s))
        return data


def normalize_reference(data: Any) -> List[dict]:
    """
    Normalize the per-metric reference JSON into a list of dicts,
    one per entry_idx in order.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # If it happens to be wrapped, try common keys
        for key in ("data", "items", "entries", "list", "captions"):
            if isinstance(data.get(key), list):
                return data[key]
        # Else assume it's a single entry
        return [data]
    # Unknown type -> empty
    return []


def build_reference_maps(
    ref_list: List[dict],
) -> Tuple[Dict[int, str], Dict[str, str], Dict[str, str]]:
    """
    Build lookup maps:
      by_idx:        entry_idx -> original_caption
      by_caption:    caption string -> original_caption
      by_framefirst: frame_first string -> original_caption
    """
    by_idx = {}
    by_caption = {}
    by_framefirst = {}

    for i, item in enumerate(ref_list):
        oc = item.get("original_caption")
        if oc is None:
            oc = item.get("caption")  # fallback if original_caption missing

        by_idx[i] = oc

        cap = item.get("caption")
        if isinstance(cap, str):
            by_caption[cap] = oc

        ff = item.get("frame_first")
        if isinstance(ff, str):
            by_framefirst[ff] = oc

    return by_idx, by_caption, by_framefirst


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_metric_dirs(metadata_root: Path) -> List[Tuple[Path, Path]]:
    """
    Return (metric_dir, relative_path) pairs for every directory that looks like
    a metric (i.e., contains caption/first_frame folders), regardless of depth.
    """
    metric_dirs: List[Tuple[Path, Path]] = []
    queue = deque([metadata_root])
    while queue:
        current = queue.popleft()
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except PermissionError:
            continue
        for child in children:
            if any((child / sub).is_dir() for sub in ("caption", "first_frame")):
                metric_dirs.append((child, child.relative_to(metadata_root)))
            else:
                queue.append(child)
    return metric_dirs


def process_metric(
    metric_dir: Path,
    rel_metric_path: Path,
    summary_root: Path,
    out_root: Path,
    strict: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Process one metric directory:
      - loads reference mapping
      - writes updated JSONs with original_caption to output tree

    Returns (processed_files, updated_files, skipped_files)
    """
    metric = metric_dir.name
    summary_file = find_summary_file(summary_root, metric)
    if summary_file is None:
        msg = f"[WARN] No summary JSON found for metric '{metric}'. Skipping."
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(msg)
        return (0, 0, 0)

    if verbose:
        print(f"[INFO] Using summary for '{metric}': {summary_file}")

    ref_raw = _load_json_loose(summary_file)
    ref_list = normalize_reference(ref_raw)
    by_idx, by_caption, by_framefirst = build_reference_maps(ref_list)

    total, updated, skipped = 0, 0, 0

    for sub in ("caption", "first_frame"):
        in_dir = metric_dir / sub
        if not in_dir.is_dir():
            continue
        out_dir = out_root / rel_metric_path / sub
        ensure_dir(out_dir)

        for jf in sorted(in_dir.glob("*.json")):
            total += 1
            try:
                obj = _load_json_loose(jf)
                # Most files should be a single dict. If a list, update each item.
                if isinstance(obj, dict):
                    oc = find_original_caption_for_item(
                        obj, by_idx, by_caption, by_framefirst
                    )
                    if oc is not None:
                        obj["original_caption"] = oc
                        updated += 1
                    else:
                        skipped += 1
                    (out_dir / jf.name).write_text(
                        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

                elif isinstance(obj, list):
                    any_updated = False
                    new_list = []
                    for it in obj:
                        if isinstance(it, dict):
                            oc = find_original_caption_for_item(
                                it, by_idx, by_caption, by_framefirst
                            )
                            if oc is not None:
                                it["original_caption"] = oc
                                any_updated = True
                            new_list.append(it)
                        else:
                            new_list.append(it)
                    if any_updated:
                        updated += 1
                    else:
                        skipped += 1
                    (out_dir / jf.name).write_text(
                        json.dumps(new_list, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                else:
                    # Unknown structure; just copy over
                    skipped += 1
                    (out_dir / jf.name).write_text(
                        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"[WARN] Failed on {jf}: {e}")

    return (total, updated, skipped)


def find_original_caption_for_item(
    item: dict,
    by_idx: Dict[int, str],
    by_caption: Dict[str, str],
    by_framefirst: Dict[str, str],
) -> Optional[str]:
    """
    Try to get original_caption for a per-file JSON object.
    Priority:
      1) entry_idx lookup (fast + reliable)
      2) exact caption match
      3) match used_text against frame_first
    """
    # 1) entry_idx
    idx = item.get("entry_idx")
    if isinstance(idx, int) and idx in by_idx:
        return by_idx[idx]

    # 2) caption string
    cap = item.get("caption")
    if isinstance(cap, str) and cap in by_caption:
        return by_caption[cap]

    # 3) used_text vs frame_first
    used_text = item.get("used_text")
    if isinstance(used_text, str) and used_text in by_framefirst:
        return by_framefirst[used_text]

    return None


def main():
    ap = argparse.ArgumentParser(description="Add original_caption to per-file JSONs.")
    ap.add_argument(
        "--metadata-root",
        required=True,
        type=Path,
        help="Root of the metrics tree with {metric}/{caption,first_frame}/*.json",
    )
    ap.add_argument(
        "--summary-root",
        required=True,
        type=Path,
        help="Root containing one per-metric reference JSON (list of entries with original_caption).",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output root where updated JSONs will be written (structure preserved).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Error out if a metric has no summary JSON.",
    )
    ap.add_argument("--verbose", action="store_true", help="Print detailed progress.")
    args = ap.parse_args()

    if not args.metadata_root.is_dir():
        raise SystemExit(f"metadata-root not found: {args.metadata_root}")
    if not args.summary_root.exists():
        raise SystemExit(f"summary-root not found: {args.summary_root}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    grand_total = grand_updated = grand_skipped = 0

    metrics = discover_metric_dirs(args.metadata_root)
    if args.verbose:
        print(
            f"[INFO] Found {len(metrics)} metric folders (nested ok) under {args.metadata_root}"
        )

    for metric_dir, rel_metric_path in metrics:
        total, updated, skipped = process_metric(
            metric_dir,
            rel_metric_path,
            args.summary_root,
            args.out_root,
            strict=args.strict,
            verbose=args.verbose,
        )
        grand_total += total
        grand_updated += updated
        grand_skipped += skipped
        if args.verbose:
            print(
                f"[DONE] {metric_dir.name}: files={total}, updated={updated}, skipped={skipped}"
            )

    print(
        f"[SUMMARY] processed={grand_total}, updated={grand_updated}, skipped={grand_skipped}"
    )
    if grand_total == 0:
        print("[NOTE] No JSON files found. Check your --metadata-root structure.")


if __name__ == "__main__":
    main()
