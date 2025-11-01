#!/usr/bin/env python3
import argparse
import ast
import json
import sys


def first_caption(value):
    # Already a proper JSON list?
    if isinstance(value, list):
        return value[0] if value else ""
    # String that looks like a list (uses single quotes) -> parse safely
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)  # handles "['a', 'b']"
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0])
            except Exception:
                pass
        return s  # fallback
    # Anything else
    return "" if value is None else str(value)


def main():
    ap = argparse.ArgumentParser(
        description="Keep only the first caption in each item."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON path")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[error] Top-level JSON must be a list of objects.", file=sys.stderr)
        sys.exit(1)

    fixed = []
    changed = 0
    for obj in data:
        if isinstance(obj, dict) and "caption" in obj:
            old = obj["caption"]
            new = first_caption(old)
            if new != old:
                changed += 1
            obj["caption"] = new
        fixed.append(obj)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(
        f"[done] Wrote {len(fixed)} items to {args.out_path}. Changed captions: {changed}"
    )


if __name__ == "__main__":
    main()
