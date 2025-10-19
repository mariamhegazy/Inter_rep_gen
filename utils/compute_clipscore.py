#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import clip  # openai-clip
import numpy as np
import torch
from tqdm import tqdm

try:
    import decord
    from decord import VideoReader
    from decord import cpu as decord_cpu
except Exception as e:
    raise ImportError("Please `pip install decord` for video reading.") from e


_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = _slug_re.sub("_", s)
    return s[:200] or "item"


def pick_caption(rec: dict, caption_field: str) -> Tuple[str, str]:
    """Return (caption, tag)."""
    if caption_field == "base":
        return (rec.get("caption") or "").strip(), "BASE"
    if caption_field == "aug":
        return (rec.get("caption_aug") or "").strip(), "AUG"
    if caption_field == "contra":
        return (rec.get("caption_contra") or "").strip(), "CONTRA"
    # auto preference: aug -> contra -> base
    for k, tag in (
        ("caption_aug", "AUG"),
        ("caption_contra", "CONTRA"),
        ("caption", "BASE"),
    ):
        v = (rec.get(k) or "").strip()
        if v:
            return v, tag
    return "", "BASE"


def load_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset_json must be a JSON list")
    return data


def build_index_by_stem(videos_dir: Path) -> Dict[str, Path]:
    idx = {}
    for p in videos_dir.rglob("*.mp4"):
        stem = p.stem
        idx[stem] = p
    return idx


def find_video_for_record(
    rec: dict, idx: Dict[str, Path], prefer_prompt_name: bool
) -> Optional[Path]:
    """
    Try to locate the video file. We try, in order:
      1) prompt-based stem (if asked to prefer it)
      2) original image stem
      3) a super-slugged prompt stem fallback
    """
    # build candidate stems
    # original stem from file_name
    file_stem = slugify(Path((rec.get("file_name") or "item")).stem)
    # prompt stems (any caption field present)
    stems = []
    for k in ("caption_aug", "caption_contra", "caption"):
        v = (rec.get(k) or "").strip()
        if v:
            stems.append(slugify(v))
    # prefer prompt names?
    ordered = (stems + [file_stem]) if prefer_prompt_name else ([file_stem] + stems)
    for s in ordered:
        if s in idx:
            return idx[s]
    return None


@torch.no_grad()
def embed_text(model, preprocess_device, text: str) -> torch.Tensor:
    tokens = clip.tokenize([text]).to(preprocess_device)
    txt_feat = model.encode_text(tokens)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return txt_feat[0]  # (d,)


@torch.no_grad()
def embed_frames(
    model, preprocess, device, vr: VideoReader, num_samples=16, out_size=224
) -> torch.Tensor:
    """Uniformly sample frames -> (K, d) normalized features."""
    n = len(vr)
    if n == 0:
        return torch.empty(0, model.visual.output_dim, device=device)

    # choose indices
    if num_samples >= n:
        inds = np.arange(n)
    else:
        # uniform
        step = n / num_samples
        inds = [int(i * step) for i in range(num_samples)]

    frames = vr.get_batch(inds)  # (K, H, W, 3) uint8
    # preprocess each frame with CLIP transform
    feats = []
    for i in range(frames.shape[0]):
        img = frames[i].asnumpy()  # HWC uint8
        pil = (
            preprocess.transforms[0](img) if hasattr(preprocess, "transforms") else None
        )  # not used
        # Use CLIP's standard preprocess pipeline
        tensor = (
            preprocess(Image_from_ndarray(img)).unsqueeze(0).to(device)
        )  # 1x3x224x224
        f = model.encode_image(tensor)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f[0])
    return torch.stack(feats, dim=0)  # (K, d)


# Minimal helper to get PIL Image from ndarray without importing PIL globally
from PIL import Image


def Image_from_ndarray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def cosine100(a: torch.Tensor, b: torch.Tensor) -> float:
    # a: (..., d), b: (d,)
    sim = a @ b  # (...,)
    val = float(sim.mean().clamp(min=0.0) * 100.0)
    return val


def cosine100_max(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = a @ b  # (...,)
    val = float(sim.max().clamp(min=0.0) * 100.0)
    return val


def main():
    ap = argparse.ArgumentParser("Compute CLIPScore on videos")
    ap.add_argument(
        "--videos_dir",
        required=True,
        help="Folder containing the .mp4s (e.g., .../TI2V/BASE)",
    )
    ap.add_argument(
        "--dataset_json", required=True, help="JSON list (original or aug/contra)"
    )
    ap.add_argument(
        "--caption_field", default="auto", choices=["auto", "base", "aug", "contra"]
    )
    ap.add_argument(
        "--prefer_prompt_filenames",
        action="store_true",
        help="Try to match videos saved with prompt-as-name first",
    )
    ap.add_argument(
        "--model", default="ViT-L/14", help="CLIP model name (e.g., ViT-B/32, ViT-L/14)"
    )
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--frame_samples", type=int, default=16)
    ap.add_argument("--outfile_csv", default="clipscore_video.csv")
    ap.add_argument("--outfile_json", default="clipscore_video.json")
    args = ap.parse_args()

    device = (
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )

    # Load CLIP
    model, preprocess = clip.load(args.model, device=device)  # opensource CLIP
    model.eval()

    # Build video index
    videos_dir = Path(args.videos_dir)
    vid_index = build_index_by_stem(videos_dir)

    # Load dataset
    data = load_json_list(args.dataset_json)

    results = []
    miss = 0
    pbar = tqdm(data, desc="CLIPScore(videos)")
    for rec in pbar:
        cap, tag = pick_caption(rec, args.caption_field)
        if not cap:
            continue
        vid_path = find_video_for_record(rec, vid_index, args.prefer_prompt_filenames)
        if vid_path is None:
            miss += 1
            continue

        # Read frames
        try:
            vr = VideoReader(str(vid_path), ctx=decord_cpu(0))
        except Exception as e:
            # Bad file – skip but record
            results.append(
                {
                    "file_name": rec.get("file_name", ""),
                    "caption": cap,
                    "cap_tag": tag,
                    "video_path": str(vid_path),
                    "clip_mean": None,
                    "clip_max": None,
                    "error": f"decord_open_fail: {repr(e)}",
                }
            )
            continue

        # Embeddings
        txt = embed_text(model, device, cap)
        frm = embed_frames(
            model, preprocess, device, vr, num_samples=args.frame_samples
        )
        if frm.numel() == 0:
            # empty video somehow
            results.append(
                {
                    "file_name": rec.get("file_name", ""),
                    "caption": cap,
                    "cap_tag": tag,
                    "video_path": str(vid_path),
                    "clip_mean": None,
                    "clip_max": None,
                    "error": "no_frames",
                }
            )
            continue

        s_mean = cosine100(frm, txt)
        s_max = cosine100_max(frm, txt)
        results.append(
            {
                "file_name": rec.get("file_name", ""),
                "caption": cap,
                "cap_tag": tag,
                "video_path": str(vid_path),
                "clip_mean": round(s_mean, 4),
                "clip_max": round(s_max, 4),
            }
        )

    # Aggregate
    vals_mean = [
        r["clip_mean"] for r in results if isinstance(r.get("clip_mean"), (float, int))
    ]
    vals_max = [
        r["clip_max"] for r in results if isinstance(r.get("clip_max"), (float, int))
    ]
    agg = {
        "count": len(results),
        "missing_videos": miss,
        "mean_of_means": float(np.mean(vals_mean)) if vals_mean else None,
        "mean_of_maxes": float(np.mean(vals_max)) if vals_max else None,
    }

    # Save CSV
    fieldnames = [
        "file_name",
        "caption",
        "cap_tag",
        "video_path",
        "clip_mean",
        "clip_max",
        "error",
    ]
    with open(args.outfile_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)

    # Save JSON
    out_json = {"aggregate": agg, "items": results}
    with open(args.outfile_json, "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print(f"[CLIPScore] videos: {len(results)}, missing: {miss}")
    print(
        f"[CLIPScore] mean_of_means: {agg['mean_of_means']}, mean_of_maxes: {agg['mean_of_maxes']}"
    )
    print(f"[CLIPScore] wrote: {args.outfile_csv} and {args.outfile_json}")


if __name__ == "__main__":
    main()


# python utils/compute_clipscore.py \
#   --videos_dir samples/wan-videos-vbench/TI2V \
#   --dataset_json /capstor/store/cscs/swissai/a144/mariam/vbench2_beta_i2v/data/i2v-bench-info.json \
#   --caption_field base \
#   --prefer_prompt_filenames \
#   --outfile_csv ti2v_base_clip.csv \
#   --outfile_json ti2v_base_clip.json

# # T2V with AUG captions (your augmented json)
# python utils/compute_clipscore.py \
#   --videos_dir /path/to/out/T2V/AUG \
#   --dataset_json prompts_augmented.json \
#   --caption_field aug \
#   --prefer_prompt_filenames \
#   --outfile_csv t2v_aug_clip.csv \
#   --outfile_json t2v_aug_clip.json
