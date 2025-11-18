#!/usr/bin/env python3
# reorganize_videos_by_reference.py
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def build_reference(src_root: Path):
    """
    Build a mapping of filename -> list of relative parent dirs where it appears in src.
    Returns:
      - ref_map: dict[str, list[Path]] (relative paths)
      - total_ref: total count of video files in src tree
    """
    ref_map = defaultdict(list)
    total = 0
    for p in src_root.rglob("*"):
        if p.is_file() and is_video(p):
            rel_parent = p.relative_to(src_root).parent  # subfolder path
            ref_map[p.name].append(rel_parent)
            total += 1
    return ref_map, total


def index_pool(pool_dir: Path):
    """
    Index flat videos in pool_dir by filename -> absolute path.
    If duplicates exist in the pool, keep the first and note others.
    """
    index = {}
    dups = defaultdict(list)
    for p in pool_dir.iterdir():
        if p.is_file() and is_video(p):
            if p.name in index:
                dups[p.name].append(p)
            else:
                index[p.name] = p
    return index, dups


def main():
    ap = argparse.ArgumentParser(
        description="Mirror subfolders from src_root and move/copy flat videos from pool into matching subfolders in dest_root."
    )
    ap.add_argument(
        "--src_root",
        required=True,
        type=Path,
        help="Structured reference tree (with subfolders).",
    )
    ap.add_argument(
        "--dest_root",
        required=True,
        type=Path,
        help="Destination root (subfolders will be created here).",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="Folder containing the flat videos. Defaults to dest_root.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview actions without moving/copying/removing files.",
    )
    args = ap.parse_args()

    src_root: Path = args.src_root.resolve()
    dest_root: Path = args.dest_root.resolve()
    pool_dir: Path = (args.pool or args.dest_root).resolve()

    if not src_root.exists():
        raise SystemExit(f"src_root not found: {src_root}")
    if not dest_root.exists():
        print(f"[info] Creating dest_root: {dest_root}")
        if not args.dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
    if not pool_dir.exists():
        raise SystemExit(f"pool directory not found: {pool_dir}")

    # 1) Scan reference tree -> filename -> [rel_parent1, rel_parent2, ...]
    ref_map, total_ref = build_reference(src_root)

    # 2) Index pool (flat videos)
    pool_index, pool_dups = index_pool(pool_dir)
    if pool_dups:
        print(
            f"[warn] Pool has {len(pool_dups)} duplicate filenames; the first found will be used."
        )

    moved, copied, skipped_existing = 0, 0, 0
    missing_paths = []  # collect full relative target paths for missing files

    for fname, rel_parents in sorted(ref_map.items()):
        src_in_pool = pool_index.get(fname)

        # Compute all target paths (relative to dest_root)
        targets = [
            (rel_parent, (dest_root / rel_parent / fname)) for rel_parent in rel_parents
        ]

        if src_in_pool is None:
            # Collect all expected target paths for this missing filename
            for rel_parent, target_path in targets:
                # store as relative path for readability
                rel_path = (
                    (rel_parent / fname).as_posix() if str(rel_parent) != "." else fname
                )
                missing_paths.append(rel_path)
            # (Optional) also print per-file missing notice
            print(f"[missing] Not found in pool: {fname}")
            continue

        # Ensure target dirs exist
        for rel_parent, target_path in targets:
            if not args.dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)

        if len(targets) == 1:
            # Unique occurrence -> MOVE
            target_path = targets[0][1]
            if target_path.exists():
                skipped_existing += 1
                print(f"[skip] Already exists: {target_path.relative_to(dest_root)}")
            else:
                print(f"[move] {src_in_pool}  ->  {target_path}")
                if not args.dry_run:
                    try:
                        shutil.move(src_in_pool.as_posix(), target_path.as_posix())
                    except Exception as e:
                        print(f"[error] move failed: {e}")
                        continue
                moved += 1
            pool_index.pop(fname, None)
        else:
            # Duplicate in src -> COPY to each, then delete original from pool
            any_copied = False
            for rel_parent, target_path in targets:
                if target_path.exists():
                    skipped_existing += 1
                    print(
                        f"[skip] Already exists: {target_path.relative_to(dest_root)}"
                    )
                    continue
                print(f"[copy] {src_in_pool}  ->  {target_path}")
                if not args.dry_run:
                    try:
                        shutil.copy2(src_in_pool.as_posix(), target_path.as_posix())
                    except Exception as e:
                        print(f"[error] copy failed: {e}")
                        continue
                copied += 1
                any_copied = True

            if any_copied and not args.dry_run:
                try:
                    src_in_pool.unlink(missing_ok=True)
                    pool_index.pop(fname, None)
                except Exception as e:
                    print(
                        f"[warn] Could not remove original from pool: {src_in_pool} ({e})"
                    )

    print("\n--- Summary ---")
    print(f"Reference videos (total entries in src tree): {total_ref}")
    print(f"Moved (unique filenames):                    {moved}")
    print(f"Copied (for duplicates in src):              {copied}")
    print(f"Skipped (already existed at target):         {skipped_existing}")
    print(f"Missing in pool (count):                     {len(missing_paths)}")

    if missing_paths:
        print("\n--- Missing target paths (relative to dest_root) ---")
        for rel in sorted(set(missing_paths)):
            print(rel)

    if args.dry_run:
        print("\n(dry run: no files moved/copied/removed)")


if __name__ == "__main__":
    main()
