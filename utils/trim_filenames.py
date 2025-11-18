#!/usr/bin/env python3
import argparse
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".mpg", ".mpeg", ".m4v", ".gif"}


def safe_rename(src: Path, dst_name: str, max_len: int) -> Path:
    """
    Rename src to dst_name, resolving collisions by appending _1, _2, ...
    Ensures final name length <= max_len. Returns final path.
    """
    dst = src.with_name(dst_name)
    if dst == src:
        return src

    # If available, rename directly
    if not dst.exists():
        src.rename(dst)
        return dst

    # Otherwise, append suffixes while respecting max_len
    stem = dst.stem
    ext = dst.suffix
    i = 1
    while True:
        suffix = f"_{i}"
        # Make sure name length stays within max_len
        allowed_stem_len = max_len - len(ext) - len(suffix)
        trimmed_stem = stem[:allowed_stem_len] if allowed_stem_len > 0 else ""
        candidate = src.with_name(f"{trimmed_stem}{suffix}{ext}")
        if not candidate.exists():
            src.rename(candidate)
            return candidate
        i += 1


def main():
    p = argparse.ArgumentParser(description="Trim long video filenames in a folder.")
    p.add_argument("folder", type=Path, help="Folder with videos (non-recursive).")
    p.add_argument(
        "--max-length",
        type=int,
        default=180,
        help="Max filename length (default: 180).",
    )
    p.add_argument(
        "--exts",
        nargs="*",
        default=list(VIDEO_EXTS),
        help="Video extensions to process (default: common video types).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without renaming.",
    )
    p.add_argument(
        "--literal",
        action="store_true",
        help="Trim the entire filename (including extension) to max length. "
        "By default only the base name is trimmed and the extension is preserved outside the limit.",
    )
    args = p.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}
    changed = 0
    skipped = 0

    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue

        name = f.name
        if len(name) <= args.max_length:
            skipped += 1
            continue

        if args.literal:
            # Cut whole filename to max_length
            new_name = name[: args.max_length]
            # If we cut away the extension entirely, that's intentional in literal mode.
            # But avoid empty name.
            if not new_name:
                continue
        else:
            # Preserve extension; trim only the stem within the limit
            ext = f.suffix
            stem = f.stem
            allowed_stem_len = max(args.max_length - len(ext), 0)
            new_stem = stem[:allowed_stem_len]
            if not new_stem:  # ensure something remains before the extension
                new_stem = "_"
            new_name = f"{new_stem}{ext}"

        if new_name == name:
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY] {name}  ->  {new_name}")
            changed += 1
        else:
            final_path = safe_rename(f, new_name, args.max_length)
            print(f"[OK ] {name}  ->  {final_path.name}")
            changed += 1

    print(f"\nDone. Renamed: {changed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
