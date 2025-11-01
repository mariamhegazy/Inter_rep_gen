# parallel_panda_download.py
# pip install yt-dlp pandas tqdm

import argparse
import ast
import os
import re
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def to_seconds(hms: str) -> float:
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        raise ValueError(f"Bad time format: {hms}")
    return h * 3600 + m * 60 + s


def pick_first_window(ts_field):
    """
    ts_field looks like: "[['0:00:19.227','0:00:28.278'], ['...','...']]"
    Return (start_sec, end_sec) for the first window, or (None, None).
    """
    if pd.isna(ts_field) or not str(ts_field).strip():
        return None, None
    try:
        windows = ast.literal_eval(ts_field)
        if not windows:
            return None, None
        start, end = windows[0]
        return to_seconds(str(start)), to_seconds(str(end))
    except Exception:
        return None, None


def fmt_hms(t: float) -> str:
    # hh:mm:ss.mmm
    return f"{int(t//3600)}:{int((t%3600)//60):02d}:{(t%60):06.3f}"


def build_out_name(url: str, vid_field: str | None, idx: int) -> str:
    if vid_field:
        base = str(vid_field).strip()
        if base:
            return base
    # fallback: try to extract YouTube ID
    m = re.search(r"[?&]v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    return f"clip_{idx}"


def download_one(row_tuple):
    i, row, out_dir, col_url, col_vid, col_ts = row_tuple
    url = str(row[col_url]).strip()
    if not url or not url.startswith("http"):
        return (i, False, "bad_url")

    base = build_out_name(
        url, (row[col_vid] if col_vid and pd.notna(row[col_vid]) else None), i
    )
    out_path = os.path.join(out_dir, f"{base}.mp4")
    if os.path.exists(out_path):
        return (i, True, "exists")

    start, end = (None, None)
    if col_ts:
        start, end = pick_first_window(row[col_ts])

    cmd = [
        "yt-dlp",
        "-f",
        "mp4",
        "--no-warnings",
        "--concurrent-fragments",
        "4",  # can help on some hosts
        "-R",
        "10",  # retries
        "--fragment-retries",
        "10",
        "--continue",
        "-o",
        out_path,
        url,
    ]

    if start is not None and end is not None and end > start:
        cmd += ["--download-sections", f"*{fmt_hms(start)}-{fmt_hms(end)}"]

    # If you hit age/region gates, uncomment and point to your cookies:
    # cmd += ["--cookies", "/path/to/cookies.txt"]

    try:
        # Use text mode so yt-dlp can show errors cleanly if any
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        ok = (result.returncode == 0) and os.path.exists(out_path)
        return (i, ok, "ok" if ok else f"fail:{result.stderr.strip()[:200]}")
    except Exception as e:
        return (i, False, f"exc:{e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/capstor/store/cscs/swissai/a144/datasets/panda-70m/ppanda70m_training_2m.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/capstor/store/cscs/swissai/a144/datasets/panda-70m/validation_videos",
    )
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10_000,
        help="Download this many random rows (max).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Make Ctrl-C stop cleanly
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    df = pd.read_csv(args.csv)

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    col_url = cols.get("url") or cols.get("video_url")
    col_vid = cols.get("videoid") or cols.get("video_id")
    col_ts = cols.get("timestamp")
    if not col_url:
        raise RuntimeError(f"Could not find URL column among {list(df.columns)}")

    # Keep only rows with http URLs, then sample up to n_samples
    df = df[df[col_url].astype(str).str.startswith("http", na=False)]
    if len(df) == 0:
        raise RuntimeError("No valid http URLs found in the CSV.")
    if args.n_samples > 0 and len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    tasks = [
        (i, row, args.out_dir, col_url, col_vid, col_ts) for i, row in df.iterrows()
    ]

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(download_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            _, ok, _ = f.result()
            if ok:
                successes += 1
            else:
                failures += 1

    print(
        f"\n✅ Done. Success: {successes}, Failures: {failures}, Total attempted: {len(tasks)}"
    )


if __name__ == "__main__":
    main()
