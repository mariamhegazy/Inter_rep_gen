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


def load_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


# ---- Get sorted JSON lists ----
single_files = sorted(glob(os.path.join(single_animal_dir, "*.json")))
full_files = sorted(glob(os.path.join(full_prompt_dir, "*.json")))

# Sanity check
if len(single_files) != len(full_files):
    print(
        f"⚠️  Warning: {len(single_files)} single JSONs vs {len(full_files)} full-prompt JSONs (using min count)."
    )

# ---- Replace captions sequentially ----
n_files = min(len(single_files), len(full_files))
total_updated = 0

for i in range(n_files):
    sfile = single_files[i]
    ffile = full_files[i]

    single_data = load_json_file(sfile)
    full_data = load_json_file(ffile)

    # replace sequentially within file
    n_entries = min(len(single_data), len(full_data))
    for j in range(n_entries):
        single_data[j]["caption"] = full_data[j]["caption"]
        total_updated += 1

    out_path = os.path.join(output_dir, os.path.basename(sfile))
    with open(out_path, "w") as f:
        json.dump(single_data, f, indent=2)

    print(f"[updated] {os.path.basename(sfile)} ({n_entries} captions replaced)")

print(f"\n✅ Done! Replaced {total_updated} captions sequentially.")
print(f"Saved updated JSONs in: {output_dir}")
