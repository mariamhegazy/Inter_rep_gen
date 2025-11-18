#!/usr/bin/env python3
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


def clean_stem(stem: str) -> str:
    """
    Remove a trailing pattern like [._-]idx<digits>_r<digits> from the stem.
    Examples:
      '...Garden._idx00064_r0'  -> '...Garden'
      'name-idx12_r3'           -> 'name'
    We also strip trailing spaces/dots/underscores/hyphens left behind.
    """
    m = re.search(r"(?i)^(?P<base>.*?)(?:[._-]?idx\d+_r\d+)$", stem)
    base = m.group("base") if m else stem
    base = base.rstrip(" ._-")
    return base if base else "_"


def main():
    parser = argparse.ArgumentParser(
        description="Combine seed folders per model and rename videos with suffix _0, _1, ... by sorted seed, "
        "stripping trailing _idxNNNNN_rM tokens."
    )
    parser.add_argument(
        "root", type=Path, help="Root folder containing the seed folders"
    )
    parser.add_argument(
        "--subdir",
        default="T2V/BASE/Diversity",
        help="Subdirectory inside each seed folder where the videos are located (default: T2V/BASE/Diversity)",
    )
    parser.add_argument(
        "--dest-suffix",
        default="_combined",
        help="Suffix for destination folder per model (default: _combined)",
    )
    parser.add_argument(
        "--exts",
        nargs="*",
        default=[".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"],
        help="Video extensions to include (default: common video types)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without copying files"
    )
    args = parser.parse_args()

    seed_dir_re = re.compile(r"^(?P<model>.+)-seed_(?P<seed>\d+)$")
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    # Collect seed folders grouped by model
    models = {}  # model -> list[(seed_int, seed_path)]
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        m = seed_dir_re.match(d.name)
        if not m:
            continue
        model = m.group("model")
        seed = int(m.group("seed"))
        models.setdefault(model, []).append((seed, d))

    if not models:
        raise SystemExit("No folders matching '*-seed_<num>' were found.")

    exts = {e.lower() for e in args.exts}

    def find_videos_in(dirpath: Path):
        """Return {stem: Path} for video files directly under dirpath."""
        if not dirpath.exists():
            return {}
        vmap = {}
        for p in dirpath.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                vmap[p.stem] = p
        return vmap

    for model, seed_entries in sorted(models.items()):
        # Sort seeds numerically for stable indexing _0, _1, ...
        seed_entries.sort(key=lambda x: x[0])
        dest_dir = root / f"{model}{args.dest_suffix}"
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        seed_maps = []
        for seed, seed_root in seed_entries:
            src_dir = seed_root / args.subdir
            vmap = find_videos_in(src_dir)
            if not vmap:
                print(
                    f"[WARN] No videos found at {src_dir} (model={model}, seed={seed})"
                )
            seed_maps.append((seed, src_dir, vmap))

        # If nothing at all, skip
        if all(len(vmap) == 0 for _, _, vmap in seed_maps):
            print(f"[WARN] Skipping model '{model}' (no videos detected in any seed).")
            continue

        # Use intersection of stems across seeds so each video has the same set of instances
        non_empty_maps = [vmap for _, _, vmap in seed_maps if vmap]
        common_stems = set(non_empty_maps[0].keys())
        for vmap in non_empty_maps[1:]:
            common_stems &= set(vmap.keys())

        if not common_stems:
            print(
                f"[WARN] No common video filenames across seeds for model '{model}'. Skipping."
            )
            continue

        # For collision safety after cleaning (different originals may collapse to same base)
        dedup_counter = defaultdict(int)

        print(
            f"[INFO] Model '{model}': {len(seed_entries)} seeds, {len(common_stems)} videos per seed."
        )
        for orig_stem in sorted(common_stems):
            base = clean_stem(orig_stem)
            dedup_counter[base] += 1
            # If this base name has appeared before for a different original, disambiguate
            unique_base = (
                base if dedup_counter[base] == 1 else f"{base}__{dedup_counter[base]}"
            )

            for idx, (seed, src_dir, vmap) in enumerate(seed_maps):
                src = vmap.get(orig_stem)
                if not src:
                    print(
                        f"[WARN] Missing '{orig_stem}' in {src_dir}; skipping this instance."
                    )
                    continue

                dst = dest_dir / f"{unique_base}_{idx}{src.suffix}"
                if dst.exists() and not args.overwrite:
                    print(f"[SKIP] Exists: {dst} (use --overwrite to replace)")
                    continue
                action = "COPY" if not dst.exists() else "OVERWRITE"
                print(f"[{action}] {src} -> {dst}")
                if not args.dry_run:
                    shutil.copy2(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
