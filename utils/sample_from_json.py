#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path


def sample_from_json_array(path, k, seed):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list (array).")
    random.seed(seed)
    return data if k >= len(data) else random.sample(data, k)


def sample_from_jsonl(path, k, seed):
    random.seed(seed)
    sample, n = [], 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if len(sample) < k:
                sample.append(obj)
            else:
                j = random.randint(1, n)
                if j <= k:
                    sample[j - 1] = obj
    return sample


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        chunk = f.read(4096)
    for ch in chunk:
        if not ch.isspace():
            return "array" if ch == "[" else "jsonl"
    return "array"


def main():
    ap = argparse.ArgumentParser(
        description="Randomly sample K entries from a JSON dataset."
    )
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training_summarized.json"
        ),
        help="Path to input .json or .jsonl",
    )
    ap.add_argument(
        "-k",
        "--count",
        type=int,
        default=5000,
        help="Number of entries to sample (default: 5000)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="PRNG seed for reproducibility"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training_summarized_5k.json"
        ),
        help="Output path (default: <input>.sampleK.json in same folder)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of writing a new one",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    fmt = detect_format(args.input)
    picked = (
        sample_from_json_array(args.input, args.count, args.seed)
        if fmt == "array"
        else sample_from_jsonl(args.input, args.count, args.seed)
    )

    if args.overwrite:
        out_path = args.input
    else:
        out_path = args.output
        if out_path is None:
            stem = args.input.stem
            ext = "".join(args.input.suffixes) or ".json"
            out_ext = ".jsonl" if ext.endswith(".jsonl") else ".json"
            out_path = args.input.with_name(f"{stem}.sample{len(picked)}{out_ext}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if fmt == "array" or out_path.suffix == ".json":
            json.dump(picked, f, indent=2, ensure_ascii=False)
        else:
            for obj in picked:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Sampled {len(picked)} entries → {out_path}")


if __name__ == "__main__":
    main()
