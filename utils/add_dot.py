#!/usr/bin/env python3
import argparse
import re
import uuid
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif", ".m4v", ".mpg", ".mpeg"}

# Match names like "<base>_<digits><ext>" where base does NOT end with '.'
# Examples: "Garden_1.mp4" -> base="Garden", idx="1", ext=".mp4"
PATTERN = re.compile(r"^(?P<base>.*?)(?<!\.)_(?P<idx>\d+)(?P<ext>\.[^.]+)$")


def stage_temp(plans, dry_run):
    staged = []
    for src, final_name in plans:
        tmp = src.with_name(src.name + f".fixdot.tmp.{uuid.uuid4().hex}")
        if dry_run:
            print(f"[DRY] STAGE  {src.name} -> {tmp.name}")
        else:
            src.rename(tmp)
        staged.append((tmp, final_name, src.name))
    return staged


def finalize(staged, overwrite, dry_run):
    for tmp, final_name, orig_name in staged:
        dst = tmp.with_name(final_name)
        if dst.exists() and not overwrite:
            # restore original name if we can't overwrite
            if dry_run:
                print(
                    f"[DRY] SKIP (exists) {dst.name}; RESTORE {tmp.name} -> {orig_name}"
                )
            else:
                tmp.rename(tmp.with_name(orig_name))
            continue
        if dry_run:
            action = "OVERWRITE" if dst.exists() else "RENAME"
            print(f"[DRY] {action} {tmp.name} -> {dst.name}")
        else:
            if dst.exists():
                dst.unlink()
            tmp.rename(dst)
            print(f"[OK ] {orig_name} -> {dst.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Add a dot before the index: Base_0.ext -> Base._0.ext (non-recursive)."
    )
    ap.add_argument("folder", type=Path, help="Folder containing videos")
    ap.add_argument(
        "--exts",
        nargs="*",
        default=list(VIDEO_EXTS),
        help="Extensions to process (default: common video types)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview changes only")
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite if destination exists"
    )
    ap.add_argument(
        "--include-nonvideo",
        action="store_true",
        help="Also process files with other extensions",
    )
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    plans = []
    for p in sorted(args.folder.iterdir()):
        if not p.is_file():
            continue
        if not args.include_nonvideo and p.suffix.lower() not in exts:
            continue
        m = PATTERN.match(p.name)
        if not m:
            continue
        base = m.group("base")
        idx = m.group("idx")
        ext = m.group("ext")
        new_name = f"{base}._{idx}{ext}"
        if new_name == p.name:
            continue
        print(f"PLAN: {p.name} -> {new_name}")
        plans.append((p, new_name))

    if not plans:
        print("Nothing to rename.")
        return

    staged = stage_temp(plans, args.dry_run)
    finalize(staged, overwrite=args.overwrite, dry_run=args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
