#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert two-animal JSON to action-binding format."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs/Two_animals_experiments/two_animals_jumping_rope_watching/sample_0000.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="meta_data_animals/two_animals_jumping_rope_watching_action_binding.json",
        help="Path to save the formatted JSON file.",
    )
    args = parser.parse_args()

    # Load input JSON
    with open(args.input_json, "r") as f:
        data = json.load(f)

    formatted = []
    pattern = re.compile(
        r"([Aa]n?\s+[\w\s-]+)\s+jumping\s+rope\s+and\s+([Aa]n?\s+[\w\s-]+)\s+watching",
        re.IGNORECASE,
    )

    for item in data:
        caption = item["caption"].strip()
        m = pattern.search(caption)
        if not m:
            print(f"[warn] Could not parse caption: {caption}")
            continue

        subj1, subj2 = m.groups()
        subj1_clean = subj1.lower().strip()
        subj2_clean = subj2.lower().strip()

        entry = {
            "caption": caption,
            "phrase_0": [f"{subj1_clean}?", f"{subj1_clean} jumping rope?"],
            "phrase_1": [f"{subj2_clean}?", f"{subj2_clean} watching?"],
        }
        formatted.append(entry)

    # Save output JSON
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"[ok] Saved {len(formatted)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
