#!/usr/bin/env python3
import json
import os
from glob import glob

# -------- CONFIG --------
single_animal_dir = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs/Two_animals_experiments/single_animal"
full_prompt_dir = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs/Two_animals_experiments/two_animals_fight_full_prompt"
output_dir = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs/mismatch_imgs_full_prompt/single_animal_with_full_prompts"
os.makedirs(output_dir, exist_ok=True)
# ------------------------


def load_jsons(folder):
    """Return a dict mapping filename -> JSON content (list or dict)."""
    json_files = sorted(glob(os.path.join(folder, "*.json")))
    out = {}
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        out[os.path.basename(jf)] = data
    return out


def normalize_path(p):
    """Make path comparable: lowercase and strip subfolders."""
    return os.path.basename(p).lower()


# --- Load both folders ---
single_jsons = load_jsons(single_animal_dir)
full_jsons = load_jsons(full_prompt_dir)

# --- Build lookup by file basename (so every image gets its matching caption) ---
full_lookup = {}
for data in full_jsons.values():
    items = data if isinstance(data, list) else [data]
    for entry in items:
        if "file_name" in entry:
            key = normalize_path(entry["file_name"])
            full_lookup[key] = entry["caption"]

print(f"[info] Built lookup for {len(full_lookup)} full-prompt entries.")

# --- Update all entries in single_jsons ---
updated_count = 0
missing_count = 0

for fname, data in single_jsons.items():
    items = data if isinstance(data, list) else [data]
    for entry in items:
        key = normalize_path(entry["file_name"])
        if key in full_lookup:
            entry["caption"] = full_lookup[key]
            updated_count += 1
        else:
            missing_count += 1
            print(f"[warn] No match found for {key}")
    # Save to output folder (same filename)
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[updated] {fname}")

print(f"\n✅ Done! Updated captions for {updated_count} entries.")
if missing_count:
    print(f"⚠️  {missing_count} entries had no matching caption in full-prompt JSONs.")
print(f"Saved updated JSONs in: {output_dir}")
