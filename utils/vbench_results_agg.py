#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def is_number(x):
    try:
        # also handle numpy numeric types
        import numbers

        return isinstance(x, numbers.Number) and not isinstance(x, bool)
    except Exception:
        return False


def load_overall_scores(path: str, exclude_dims: set) -> Dict[str, float]:
    """
    Load a VBench/VBench-2.0 evaluation json and extract an overall score per dimension.

    NEW (preferred): expects each dimension to look like
      {
        "mean_of_means": ...,
        "std_of_means": ...,
        "overall_mean_min": ...,
        "overall_mean_max": <WE USE THIS>,
        "overall_mean_mean": ...,
        "n_runs": ...,
        "per_video": {...}
      }

    Legacy fallback: each dimension is [overall_score, [... per-video list ...]]

    Returns {dimension_name: overall_score} using "overall_mean_max" when available,
    skipping any dimensions listed in exclude_dims.
    """
    with open(path, "r") as f:
        data = json.load(f)

    out: Dict[str, float] = {}
    for dim, payload in data.items():
        if dim in exclude_dims:
            continue

        val = None

        # Preferred: dict with "overall_mean_max"
        if isinstance(payload, dict):
            if "overall_mean_max" in payload and is_number(payload["overall_mean_max"]):
                val = float(payload["overall_mean_max"])
            # extra fallback variants if present in some files
            elif "overall" in payload and is_number(payload["overall"]):
                val = float(payload["overall"])

        # Legacy: list/tuple [overall, per-video...]
        elif (
            isinstance(payload, (list, tuple))
            and len(payload) >= 1
            and is_number(payload[0])
        ):
            val = float(payload[0])

        if val is not None and not math.isnan(val):
            out[dim] = val

    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_runs(
    runs: List[Tuple[str, Dict[str, float]]],
    constants: Dict[str, Dict[str, float]] | None,
) -> Tuple[List[Tuple[str, Dict[str, float]]], Dict[str, Tuple[float, float]]]:
    """
    Normalize all runs per dimension.

    If `constants` is provided, expect constants[dim] = {"min": a, "max": b}.
    Otherwise compute global min/max across the provided runs for each dimension.

    Returns:
      normalized_runs: same structure as `runs` but with normalized values in [0,1]
      used_minmax: {dim: (min_val, max_val)} actually used for normalization
    """
    all_dims = sorted(set().union(*[set(sc.keys()) for _, sc in runs]))
    used_minmax: Dict[str, Tuple[float, float]] = {}

    if constants:
        for d in all_dims:
            c = constants.get(d)
            if c is not None and "min" in c and "max" in c:
                used_minmax[d] = (float(c["min"]), float(c["max"]))
        missing = [d for d in all_dims if d not in used_minmax]
        if missing:
            obs_minmax = compute_observed_minmax(runs, missing)
            used_minmax.update(obs_minmax)
    else:
        used_minmax = compute_observed_minmax(runs, all_dims)

    norm_runs: List[Tuple[str, Dict[str, float]]] = []
    for name, scores in runs:
        norm_scores = {}
        for d, v in scores.items():
            mn, mx = used_minmax[d]
            if mx > mn:
                norm_v = (v - mn) / (mx - mn)
            else:
                norm_v = 0.5
            norm_scores[d] = clamp01(norm_v)
        norm_runs.append((name, norm_scores))

    return norm_runs, used_minmax


def compute_observed_minmax(
    runs: List[Tuple[str, Dict[str, float]]], dims: List[str] | set[str]
) -> Dict[str, Tuple[float, float]]:
    used: Dict[str, Tuple[float, float]] = {}
    dims = list(dims)
    for d in dims:
        vals = [sc[d] for _, sc in runs if d in sc and is_number(sc[d])]
        if len(vals) == 0:
            used[d] = (0.0, 1.0)
        else:
            used[d] = (float(min(vals)), float(max(vals)))
    return used


def radar_single(
    run_name: str, scores: Dict[str, float], out_path: Path, title_suffix: str = ""
):
    if not scores:
        return
    dims = list(scores.keys())
    vals = [scores[d] for d in dims]
    dims, vals = zip(*sorted(zip(dims, vals), key=lambda x: x[0]))

    labels = list(dims) + [dims[0]]
    values = list(vals) + [vals[0]]

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)

    vmin, vmax = 0.0, 1.0
    values = [clamp01(v) for v in values]
    ax.set_ylim(vmin, vmax)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_rlabel_position(0)

    ax.plot(angles, values, linewidth=2, linestyle="-", label=run_name)
    ax.fill(angles, values, alpha=0.15)

    title = f"{run_name} — Radar" + (f" ({title_suffix})" if title_suffix else "")
    ax.set_title(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def radar_compare(
    runs: List[Tuple[str, Dict[str, float]]],
    out_path: Path,
    dims: List[str] = None,
    title: str = "Comparison Radar (Intersection, Normalized)",
):
    if not runs:
        return

    if dims is None:
        dim_sets = [set(sc.keys()) for _, sc in runs]
        shared = set.intersection(*dim_sets) if dim_sets else set()
        if not shared:
            return
        dims = sorted(shared)

    N = len(dims)
    if N == 0:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)

    vmin, vmax = 0.0, 1.0
    ax.set_ylim(vmin, vmax)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_rlabel_position(0)

    for name, scores in runs:
        vals = [clamp01(scores[d]) for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, linestyle="-", label=name)
        ax.fill(angles, vals, alpha=0.10)

    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize VBench results into compact JSON + radar charts (normalized per-dimension)."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Pairs of: NAME PATH.json (space-separated). "
            "Example: --inputs TI2V ti2v.json T2V t2v.json"
        ),
    )
    ap.add_argument(
        "--out_json", default="vbench_summary.json", help="Output JSON path"
    )
    ap.add_argument(
        "--out_dir", default="vbench_summary_plots", help="Directory for radar plots"
    )
    ap.add_argument("--title", default="VBench", help="Title for comparison radar")
    ap.add_argument(
        "--exclude-dims",
        nargs="*",
        default=None,
        help="Dimension names to exclude (default: human_action)",
    )
    ap.add_argument(
        "--constants",
        type=str,
        default="vbench_constants.json",
        help=(
            "Path to JSON with per-dimension min/max constants like: "
            '{"subject_consistency":{"min":0.35,"max":0.96}, ...}. '
            "If omitted, min/max are computed from the provided inputs."
        ),
    )
    args = ap.parse_args()

    if len(args.inputs) % 2 != 0:
        raise SystemExit("ERROR: --inputs must be NAME PATH.json pairs.")

    pairs = [
        (args.inputs[i], args.inputs[i + 1]) for i in range(0, len(args.inputs), 2)
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = set(args.exclude_dims or [])

    runs: List[Tuple[str, Dict[str, float]]] = []
    for name, path in pairs:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path} (skipping)")
            continue
        scores = load_overall_scores(path, exclude_dims=exclude)
        if not scores:
            print(f"[WARN] No valid overall scores in: {path}")
        runs.append((name, scores))

    if not runs:
        print("[WARN] No runs loaded; nothing to do.")
        return

    # Load constants if provided
    constants = None
    if args.constants:
        try:
            with open(args.constants, "r") as f:
                constants = json.load(f)
            if not isinstance(constants, dict):
                print(f"[WARN] --constants is not a dict; ignoring.")
                constants = None
        except Exception as e:
            print(f"[WARN] Failed to read --constants file: {e}")
            constants = None

    # Normalize
    norm_runs, used_minmax = normalize_runs(runs, constants)

    # Save compact JSON (both raw and normalized)
    out_payload = {
        "raw_runs": {name: sc for name, sc in runs},
        "normalized_runs": {name: sc for name, sc in norm_runs},
        "dimensions_union": sorted(set().union(*[set(sc.keys()) for _, sc in runs])),
        "dimensions_intersection": (
            sorted(set.intersection(*[set(sc.keys()) for _, sc in runs]))
            if len(runs) > 1
            else sorted(runs[0][1].keys())
        ),
        "used_minmax": {
            d: {"min": mn, "max": mx} for d, (mn, mx) in used_minmax.items()
        },
        "normalization_source": "constants" if constants else "observed",
    }
    with open(args.out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[OK] Wrote summary JSON: {args.out_json}")

    # Per-run radar (normalized)
    for name, scores in norm_runs:
        out_png = out_dir / f"radar_{name}.png"
        radar_single(name, scores, out_png, title_suffix="normalized")

    # Comparison radar on intersection (normalized)
    inter_dims = out_payload["dimensions_intersection"]
    if len(norm_runs) >= 2 and inter_dims:
        cmp_png = out_dir / "radar_comparison_intersection.png"
        radar_compare(
            norm_runs, cmp_png, dims=inter_dims, title=f"{args.title} (Normalized)"
        )

    print(f"[OK] Saved radar plots to: {out_dir}")


if __name__ == "__main__":
    main()
