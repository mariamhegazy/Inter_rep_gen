#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def collect_original_captions(category_dir: Path):
    """
    Build:
      - exact_map: file_name -> caption
      - basename_unique: basename -> caption (only if basename appears once)
    Recurse to catch JSONs in nested folders.
    """
    exact_map = {}
    basename_map = defaultdict(list)
    json_files = sorted(category_dir.rglob("*.json"))
    for jf in json_files:
        data = load_json(jf)
        if data is None:
            continue
        iterable = [data] if isinstance(data, dict) else data
        for item in iterable:
            if not isinstance(item, dict):
                continue
            fn = item.get("file_name")
            cap = item.get("caption")
            if not fn or cap is None:
                continue
            exact_map[fn] = cap
            basename_map[os.path.basename(fn)].append(cap)

    basename_unique = {b: caps[0] for b, caps in basename_map.items() if len(caps) == 1}
    return exact_map, basename_unique


def transform_item_with_original(item, fn, cat_name, exact_map, basename_unique):
    if "original_caption" in item and item["original_caption"]:
        return False  # already present

    # 1) exact path
    cap = exact_map.get(fn)

    # 2) swap '/<cat>_modified/' -> '/<cat>/'
    if cap is None:
        swapped = fn.replace(f"/{cat_name}_modified/", f"/{cat_name}/")
        cap = exact_map.get(swapped)

    # 3) unique basename fallback
    if cap is None:
        base = os.path.basename(fn)
        cap = basename_unique.get(base)

    if cap is not None:
        item["original_caption"] = cap
        return True
    return False


def update_modified_jsons_to_out(
    mod_dir: Path, out_mod_dir: Path, cat_name: str, exact_map, basename_unique
):
    """
    Read every JSON under mod_dir (recursively), add 'original_caption' when possible,
    and write the result to the mirrored path under out_mod_dir.
    """
    json_files = sorted(mod_dir.rglob("*.json"))
    added, missing, total = 0, 0, 0
    for src_path in json_files:
        rel_path = src_path.relative_to(mod_dir)
        dst_path = out_mod_dir / rel_path

        data = load_json(src_path)
        if data is None:
            # still copy the raw file so the output folder mirrors inputs
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(
                    f"[WARN] Could not copy malformed JSON {src_path} -> {dst_path}: {e}"
                )
            continue

        modified_any = False
        if isinstance(data, dict):
            items = [data]
        else:
            items = data

        for item in items:
            if not isinstance(item, dict):
                continue
            total += 1
            fn = item.get("file_name")
            if not fn:
                missing += 1
                continue
            if transform_item_with_original(
                item, fn, cat_name, exact_map, basename_unique
            ):
                added += 1
                modified_any = True
            else:
                missing += 1

        # Always save to output, even if unchanged, to keep a full mirror
        save_json(dst_path, data)

    return added, missing, total


def main():
    p = argparse.ArgumentParser(
        description="Write modified JSONs (with 'original_caption') into a new output folder, mirroring structure."
    )
    p.add_argument(
        "--base_dir",
        type=str,
        default="/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs",
        help="Folder that contains original and *_modified category folders.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs_mariam",
        help="Output root to write mirrored *_modified folders.",
    )
    args = p.parse_args()

    base = Path(args.base_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    if not base.exists() or not base.is_dir():
        print(f"[ERROR] Base dir not found: {base}")
        return
    out_root.mkdir(parents=True, exist_ok=True)

    # Map category cores to their original & modified dirs
    cats = {}
    mods = {}
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_modified"):
            core = name[:-9]  # strip suffix
            mods[core] = entry
        else:
            cats[name] = entry

    grand_added = grand_missing = grand_total = 0

    for core, mod_dir in sorted(mods.items()):
        orig_dir = cats.get(core)
        if orig_dir is None:
            print(
                f"[WARN] No original folder found for modified '{mod_dir.name}'. Skipping."
            )
            continue

        print(f"\n[INFO] Processing category: '{core}'")
        print(f"       Original: {orig_dir}")
        print(f"       Modified: {mod_dir}")

        exact_map, basename_unique = collect_original_captions(orig_dir)
        print(
            f"       Originals indexed: exact={len(exact_map)}, basename-unique={len(basename_unique)}"
        )

        out_mod_dir = out_root / mod_dir.name
        added, missing, total = update_modified_jsons_to_out(
            mod_dir, out_mod_dir, core, exact_map, basename_unique
        )
        print(f"       Wrote to: {out_mod_dir}")
        print(
            f"       Added original_caption={added}, missing matches={missing}, items seen={total}"
        )

        grand_added += added
        grand_missing += missing
        grand_total += total

    print("\n[SUMMARY]")
    print(f"  Total items seen: {grand_total}")
    print(f"  original_caption added: {grand_added}")
    print(f"  No match found: {grand_missing}")
    print(f"  Output root: {out_root}")


if __name__ == "__main__":
    main()
