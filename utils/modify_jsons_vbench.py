#!/usr/bin/env python3
import argparse
import json
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


def as_list(data):
    """Normalize JSON root to a list of dicts, return list and a flag indicating original was a dict."""
    if isinstance(data, dict):
        return [data], True
    if isinstance(data, list):
        return data, False
    return [], False


def get_ordered_captions(orig_items):
    """Return list of captions (by order). Missing captions are None."""
    caps = []
    for it in orig_items:
        caps.append(it.get("caption") if isinstance(it, dict) else None)
    return caps


def apply_order_mapping(mod_items, caps):
    """
    For each index i, set mod_items[i]['original_caption'] = caps[i] (if not None).
    DO NOT touch mod_items[i]['caption'] — keep the *_longer caption as-is.
    """
    total = len(mod_items)
    lim = min(len(mod_items), len(caps))
    added = 0
    for i in range(lim):
        if not isinstance(mod_items[i], dict):
            continue
        cap = caps[i]
        if cap is not None:
            # Always set/overwrite original_caption to ensure sync by order.
            mod_items[i]["original_caption"] = cap
            added += 1
    # Any entries beyond lim or with None caps are considered missing
    missing = total - added
    return added, missing, total


def main():
    ap = argparse.ArgumentParser(
        description="By-order mapping: keep captions from *_longer.json; add original_caption from matching original .json."
    )
    ap.add_argument(
        "--base_dir", required=True, help="Directory containing ALL JSON files (flat)."
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for processed *_longer JSONs (flat).",
    )
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    if not base.exists() or not base.is_dir():
        print(f"[ERROR] Base dir not found: {base}")
        return
    out_root.mkdir(parents=True, exist_ok=True)

    all_jsons = sorted(base.glob("*.json"))
    originals = {p.stem: p for p in all_jsons if "_longer" not in p.stem}
    modifieds = [p for p in all_jsons if "_longer" in p.stem]

    print(
        f"[INFO] Found {len(originals)} original JSONs and {len(modifieds)} *_longer JSONs in {base}"
    )

    grand_added = grand_missing = grand_total = 0

    for mod_path in modifieds:
        core = mod_path.stem.replace("_longer", "", 1)
        orig_path = originals.get(core)
        if orig_path is None:
            print(
                f"[WARN] No original JSON for {mod_path.name} (expected {core}.json). Skipping."
            )
            continue

        orig_data = load_json(orig_path)
        mod_data = load_json(mod_path)
        if orig_data is None or mod_data is None:
            print(f"[WARN] Skipping {mod_path.name} due to read error(s).")
            continue

        orig_items, _ = as_list(orig_data)
        mod_items, mod_was_dict = as_list(mod_data)

        caps = get_ordered_captions(orig_items)
        added, missing, total = apply_order_mapping(mod_items, caps)

        # Restore root shape (dict vs list)
        out_data = mod_items[0] if (mod_was_dict and mod_items) else mod_items
        dst_path = out_root / mod_path.name
        save_json(dst_path, out_data)

        print(f"[OK] {mod_path.name}: added={added}, missing={missing}, total={total}")
        grand_added += added
        grand_missing += missing
        grand_total += total

    print("\n[SUMMARY]")
    print(f"  Total items seen: {grand_total}")
    print(f"  original_caption added: {grand_added}")
    print(f"  No match found (or length mismatch remainder): {grand_missing}")
    print(f"  Output root: {out_root}")


if __name__ == "__main__":
    main()
