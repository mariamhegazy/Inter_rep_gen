#!/usr/bin/env python3
import argparse
import re
import uuid
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif", ".m4v", ".mpg", ".mpeg"}

IDX_RX = re.compile(r"(?i)(?:[._-])?idx(?P<idx>\d+)_r(?P<r>\d+)$")
CLEAN_RX = re.compile(r"(?i)^(?P<base>.*?)(?:[._-]?idx\d+_r\d+)$")


def parse_idx_r(stem: str):
    """
    Return (idx:int, r:int) if the stem contains an idx/r trailer, else None.
    """
    m = IDX_RX.search(stem)
    if not m:
        return None
    return int(m.group("idx")), int(m.group("r"))


# def clean_stem(stem: str) -> str:
#     """
#     Strip trailing 'idx######_r##' (optionally preceded by '.', '_' or '-') and tidy leftover spaces/._-.
#     """
#     m = CLEAN_RX.match(stem)
#     base = m.group("base") if m else stem
#     base = base.rstrip(" ._-")
#     return base if base else "_"


def clean_stem(stem: str) -> str:
    """
    Strip trailing 'idx######_r##' (optionally preceded by '.', '_' or '-')
    but PRESERVE any trailing '.' in the base.
    """
    CLEAN_RX = re.compile(r"(?i)^(?P<base>.*?)(?:[._-]?idx\d+_r\d+)$")
    m = CLEAN_RX.match(stem)
    base = m.group("base") if m else stem
    # keep '.', only trim spaces/underscores/hyphens
    base = base.rstrip(" _-")
    return base if base else "_"


def plan_group_renames(files):
    """
    files: list[Path] all with same cleaned base.
    Sort by (idx, r) if available, else by name.
    Return list[(src_path, final_name_str)] with final_name_str = f"{base}_{i}{ext}"
    """
    if not files:
        return []

    # All have same base & ext may differ; keep each file's own ext.
    # Derive base from any file (they share the same cleaned base by construction)
    base = clean_stem(files[0].stem)

    def sort_key(p: Path):
        pr = parse_idx_r(p.stem)
        return (0, pr[0], pr[1], p.name) if pr else (1, 0, 0, p.name)

    files_sorted = sorted(files, key=sort_key)

    plan = []
    for i, p in enumerate(files_sorted):
        plan.append((p, f"{base}_{i}{p.suffix}"))
    return plan


def stage_temp_renames(plans, dry_run: bool):
    """
    To avoid name collisions during in-place renames, first move each src to a unique temp name.
    Returns list[(temp_path, final_name_str, orig_dir)]
    """
    staged = []
    for src, final_name in plans:
        tmp = src.with_name(src.name + f".renametmp.{uuid.uuid4().hex}")
        if dry_run:
            print(f"[DRY] STAGE  {src.name} -> {tmp.name}")
        else:
            src.rename(tmp)
        staged.append((tmp, final_name, tmp.parent))
    return staged


def finalize_renames(staged, overwrite: bool, dry_run: bool):
    """
    Move each staged temp file to its final name. If overwrite is False and the
    destination exists, skip with a warning.
    """
    for tmp, final_name, directory in staged:
        dst = directory / final_name
        if dst.exists() and not overwrite:
            print(f"[SKIP] Exists: {dst.name} (use --overwrite to replace)")
            # restore original (drop temp suffix) to something sensible to avoid loss
            # best effort: keep temp as-is; user can clean manually
            continue
        if dry_run:
            action = "OVERWRITE" if dst.exists() else "RENAME"
            print(f"[DRY] {action} {tmp.name} -> {dst.name}")
        else:
            # If overwriting, unlink first to avoid cross-platform rename issues
            if dst.exists():
                dst.unlink()
            tmp.rename(dst)
            print(f"[OK ] {tmp.name} -> {dst.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Strip trailing idx######_r## and rename groups to base_0, base_1, ... within one folder."
    )
    ap.add_argument(
        "folder",
        type=Path,
        help="Folder containing the combined videos (non-recursive).",
    )
    ap.add_argument(
        "--exts",
        nargs="*",
        default=list(VIDEO_EXTS),
        help="Video extensions to process (default: common types).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing base_N files if they exist.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned operations without renaming.",
    )
    ap.add_argument(
        "--include-nonvideo",
        action="store_true",
        help="Also process files without a known video extension.",
    )
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    # Gather files that have the idx/r pattern
    candidates = []
    for p in sorted(args.folder.iterdir()):
        if not p.is_file():
            continue
        if not args.include_nonvideo and p.suffix.lower() not in exts:
            continue
        if parse_idx_r(p.stem) is None:
            # Only touch files that have the idx/r trailer
            continue
        candidates.append(p)

    if not candidates:
        print("No files with 'idx######_r##' pattern found. Nothing to do.")
        return

    # Group by cleaned base
    groups = defaultdict(list)
    for p in candidates:
        groups[clean_stem(p.stem)].append(p)

    total = 0
    all_plans = []
    for base, files in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        plans = plan_group_renames(files)
        if not plans:
            continue
        total += len(plans)
        print(
            f"[INFO] {base}: {len(plans)} file(s) -> {base}_0 .. {base}_{len(plans)-1}"
        )
        for src, final_name in plans:
            if args.dry_run:
                print(f"      {src.name}  ->  {final_name}")
        all_plans.extend(plans)

    if not all_plans:
        print("Nothing to rename after planning.")
        return

    # Two-phase in-place rename to avoid collisions
    staged = stage_temp_renames(all_plans, args.dry_run)
    finalize_renames(staged, overwrite=args.overwrite, dry_run=args.dry_run)

    print(f"\nDone. Planned/renamed: {total} file(s).")


if __name__ == "__main__":
    main()
