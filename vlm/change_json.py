#!/usr/bin/env python3
# convert_to_paraphrase_schema.py
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json", required=True, help="Path to source JSON (list of dicts)."
    )
    parser.add_argument(
        "--output_json", required=True, help="Path to write converted JSON."
    )
    parser.add_argument(
        "--start_id", type=int, default=1, help="Starting id value (default: 1)."
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    out = []
    cur_id = args.start_id
    for item in data:
        if not isinstance(item, dict):
            continue

        # Accept both "file_name" or "filename"
        file_name = item.get("file_name", item.get("filename"))

        # Required fields we expect to exist
        paraphrased = item.get("paraphrased_prompt")
        original = item.get("caption")
        typ = item.get("type")

        # If paraphrase is missing, fallback to original
        caption = paraphrased if paraphrased is not None else original

        new_entry = {
            "id": cur_id,
            "file_name": file_name,
            "type": typ,
            "caption": caption,
            "original_caption": original,
        }

        # If present in the source, pass through these optional fields
        if "caption_contra" in item:
            new_entry["caption_contra"] = item["caption_contra"]
        if "contradiction_type" in item:
            new_entry["contradiction_type"] = item["contradiction_type"]
        if "category" in item:
            new_entry["category"] = item["category"]

        out.append(new_entry)
        cur_id += 1

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(out)} items to {args.output_json}")


if __name__ == "__main__":
    main()
