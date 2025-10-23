#!/usr/bin/env python3
# evaluation/evaluate_quality_matched.py
import argparse
import difflib
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Only keep I3D helpers from your utils (we won't call its frechet_distance)
from evaluation.fvd_utils import get_fvd_logits, load_fvd_model


# ========== Repro ==========
def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_all_seeds(43)


# ========== DDP utils ==========
def ddp_print(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank{rank}] {msg}", flush=True)


def setup_ddp(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        ddp_print(
            f"Initialized DDP: rank={args.rank} local_rank={args.local_rank} world_size={args.world_size}"
        )
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        ddp_print("Running without DDP (single process)")


def cleanup_ddp():
    if dist.is_initialized():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# ========== Safe gather helpers ==========
def gather_embeddings_no_pad_mismatch(local_embs: torch.Tensor, device: str = "cuda"):
    """Original GPU-based gather (uses NCCL); returns [sum(n_i), D]."""
    if not dist.is_initialized():
        return local_embs
    world = dist.get_world_size()
    n_local = torch.tensor([local_embs.shape[0]], device=device, dtype=torch.int64)
    sizes = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(sizes, n_local)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)
    D = local_embs.shape[1]
    local_pad = local_embs
    if local_embs.shape[0] < max_n:
        pad = torch.zeros(
            max_n - local_embs.shape[0], D, device=device, dtype=local_embs.dtype
        )
        local_pad = torch.cat([local_embs, pad], dim=0)
    recv = [
        torch.zeros(max_n, D, device=device, dtype=local_embs.dtype)
        for _ in range(world)
    ]
    dist.all_gather(recv, local_pad)
    chunks = [recv[r][: sizes[r]] for r in range(world)]
    return torch.cat(chunks, dim=0)


def gather_cpu_objects(obj):
    """
    Gather arbitrary Python objects across all ranks using all_gather_object.
    Returns list of objects from each rank on every rank.
    """
    if not dist.is_initialized():
        return [obj]
    world = dist.get_world_size()
    out_list = [None for _ in range(world)]
    dist.all_gather_object(out_list, obj)
    return out_list


def concat_gathered_cpu_tensors(tensors_list):
    """
    tensors_list: list of per-rank CPU tensors (could be empty tensors).
    Returns a single CPU tensor concatenated along dim 0.
    """
    nonempty = [
        t for t in tensors_list if isinstance(t, torch.Tensor) and t.numel() > 0
    ]
    if not nonempty:
        return torch.empty(0)
    D = nonempty[0].shape[1]
    assert all(
        t.ndim == 2 and t.shape[1] == D for t in nonempty
    ), "Mismatched shapes across ranks."
    return torch.cat(nonempty, dim=0)


# ========== Files ==========
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}


def list_videos(
    root: str, recursive: bool = True, extensions: Sequence[str] | None = None
) -> List[str]:
    root = os.path.abspath(root)
    exts = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in (extensions or VIDEO_EXTS)
    }
    paths = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                paths.append(p)
    paths.sort()
    return paths


# ========== Caption → gen-filename stem (robust) ==========
_caption_end_regex = re.compile(r'"\s*,')
_multi_space_underscore = re.compile(r"[\s_]+")


def _fold_unicode(s: str) -> str:
    # fold accents, unicode quotes, etc.
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def extract_caption_text(raw_caption: str) -> str:
    """
    From samples like:
      "\"The video shows ...\",5.42,1.57,..."
    return:
      "The video shows ..."
    """
    s = raw_caption.strip()
    if s.startswith('"'):
        m = _caption_end_regex.search(s, 1)
        if m:
            return s[1 : m.start()]
        if s.endswith('"') and len(s) >= 2:
            return s[1:-1]
    parts = s.split(",")
    if len(parts) > 1:
        return parts[0]
    return s


def normalize_to_stem(caption_text: str, lowercase: bool = False, max_len: int = 240):
    """
    Convert caption text into the generated filename stem:
      - fold unicode, map non-alnum to underscore, collapse, trim.
    """
    s = _fold_unicode(caption_text.strip())
    if lowercase:
        s = s.lower()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = _multi_space_underscore.sub("_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s


def stem_tokens(stem: str) -> List[str]:
    toks = [t for t in stem.split("_") if len(t) > 1]
    return toks


def stem_signature(stem: str, first_k: int = 12) -> str:
    # A short signature used for bucketing (first K tokens)
    toks = stem_tokens(stem)
    return "_".join(toks[:first_k])


def build_gen_index(gen_paths: Sequence[str], lowercase: bool = True):
    """
    Build multiple indices:
      - stem -> path (normalized, lower if asked)
      - bucket: signature -> list of (stem, path)
      - tokens map for Jaccard
    """
    stem2path: Dict[str, str] = {}
    bucket: Dict[str, List[Tuple[str, str]]] = {}
    tokens_map: Dict[str, set] = {}

    for p in gen_paths:
        st = Path(p).stem
        st = _fold_unicode(st)
        st = re.sub(r"[^A-Za-z0-9]+", "_", st)
        st = _multi_space_underscore.sub("_", st).strip("_")
        st_key = st.lower() if lowercase else st

        stem2path[st_key] = p
        sig = stem_signature(st_key)
        bucket.setdefault(sig, []).append((st_key, p))
        tokens_map[st_key] = set(stem_tokens(st_key))
    return stem2path, bucket, tokens_map


# ========== Matching (multi-stage) ==========
def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


def match_one_stem(
    want_key: str,
    stem2path: Dict[str, str],
    bucket: Dict[str, List[Tuple[str, str]]],
    tokens_map: Dict[str, set],
    fuzzy_cutoff: float,
    min_prefix_chars: int,
    min_jaccard: float,
) -> Optional[str]:
    # 1) exact
    if want_key in stem2path:
        return stem2path[want_key]

    # candidates: same signature bucket (or a shorter signature)
    sig = stem_signature(want_key)
    candidates = bucket.get(sig, [])
    if not candidates and "_" in sig:
        short_sig = "_".join(sig.split("_")[:-1])
        candidates = bucket.get(short_sig, [])
    all_keys = list(stem2path.keys())

    # 2) prefix (for truncated filenames)
    for st_key, path in candidates:
        if len(want_key) >= min_prefix_chars and st_key.startswith(want_key):
            return path
        if len(st_key) >= min_prefix_chars and want_key.startswith(st_key):
            return path

    # 3) token jaccard
    want_tokens = tokens_map.get(want_key, set(stem_tokens(want_key)))
    best_j = 0.0
    best_path = None
    for st_key, path in candidates:
        j = jaccard(want_tokens, tokens_map.get(st_key, set()))
        if j > best_j:
            best_j, best_path = j, path
    if best_path is not None and best_j >= min_jaccard:
        return best_path

    # 4) fuzzy over candidate stems first
    cand_keys = [k for k, _ in candidates]
    if cand_keys:
        fm = difflib.get_close_matches(want_key, cand_keys, n=1, cutoff=fuzzy_cutoff)
        if fm:
            for st_key, path in candidates:
                if st_key == fm[0]:
                    return path

    # 5) fuzzy over ALL stems as last resort
    fm = difflib.get_close_matches(
        want_key, all_keys, n=1, cutoff=max(0.5, fuzzy_cutoff - 0.1)
    )
    if fm:
        return stem2path[fm[0]]
    return None


def match_pairs_from_json(
    json_items: Sequence[dict],
    gen_paths: Sequence[str],
    lowercase_match: bool = True,
    fuzzy_cutoff: float = 0.70,
    min_prefix_chars: int = 40,
    min_jaccard: float = 0.35,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[dict], List[Tuple[str, str]]]:
    """
    Returns (real_paths, gen_paths, unmatched_items, debug_pairs)
    """
    stem2gen, bucket, tokens_map = build_gen_index(gen_paths, lowercase=lowercase_match)

    real_list, gen_list = [], []
    unmatched = []
    debug_pairs = []  # (want_key, matched_key)

    for it in json_items:
        vp = it.get("video_path")
        cap_raw = it.get("caption", "")
        if not vp or not cap_raw:
            unmatched.append(
                {"video_path": vp, "caption": cap_raw, "reason": "missing_fields"}
            )
            continue

        cap_text = extract_caption_text(cap_raw)
        want_stem = normalize_to_stem(cap_text, lowercase=lowercase_match)
        want_key = want_stem.lower() if lowercase_match else want_stem

        matched_path = match_one_stem(
            want_key,
            stem2gen,
            bucket,
            tokens_map,
            fuzzy_cutoff=fuzzy_cutoff,
            min_prefix_chars=min_prefix_chars,
            min_jaccard=min_jaccard,
        )

        if matched_path:
            real_list.append(vp)
            gen_list.append(matched_path)
            debug_pairs.append((want_key, Path(matched_path).stem))
        else:
            unmatched.append(
                {
                    "video_path": vp,
                    "caption": cap_text,
                    "want_key": want_key,
                    "reason": "no_match",
                }
            )

        if limit and len(real_list) >= limit:
            break

    return real_list, gen_list, unmatched, debug_pairs


# ========== Dataset (fixed-length T; resizable storage) ==========
class MP4VideoDataset(Dataset):
    """
    Returns uint8 video array [T, H, W, C], ALWAYS length == num_frames,
    uniformly sampled; duplicates frames if the video is shorter.
    """

    def __init__(
        self, video_paths: Sequence[str], num_frames=33, target_size=(256, 256)
    ):
        self.video_paths = list(video_paths)
        self.num_frames = int(num_frames)
        self.target_size = tuple(target_size)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames, err = [], None
        try:
            with iio.imopen(video_path, "r", plugin="pyav") as reader:
                i = 0
                while True:
                    try:
                        fr = reader.read(index=i)  # RGB HxWxC
                    except IndexError:
                        break
                    frames.append(fr)
                    i += 1
        except Exception as e:
            err = e
            cap = cv2.VideoCapture(video_path)
            ok, fr = cap.read()
            while ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                ok, fr = cap.read()
            cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"No frames loaded from video: {video_path}. Original error: {err}"
            )

        # Uniform with repetition to exactly self.num_frames
        n = len(frames)
        idxs = np.clip(
            np.round(np.linspace(0, n - 1, self.num_frames)).astype(int), 0, n - 1
        )
        selected = [frames[i] for i in idxs]

        # Resize
        Ht, Wt = self.target_size
        resized = [
            cv2.resize(fr, (Wt, Ht), interpolation=cv2.INTER_LINEAR) for fr in selected
        ]
        vid = np.stack(resized, axis=0).astype(np.uint8, copy=False)
        vid = np.ascontiguousarray(vid)
        return vid


def collate_skip_none(batch):
    """Return a stack of CPU torch.uint8 tensors with resizable storage."""
    import numpy as np

    items = []
    for b in batch:
        if b is None:
            continue
        if isinstance(b, np.ndarray):
            b = np.ascontiguousarray(b)
            items.append(torch.from_numpy(b.copy()))
        elif torch.is_tensor(b):
            items.append(b.contiguous().clone())
        else:
            raise TypeError(f"Unexpected batch element type: {type(b)}")
    if not items:
        raise ValueError("Empty batch after filtering out None samples.")
    return torch.stack(items, dim=0)  # [B, T, H, W, C] on CPU


# ========== FVD (I3D) ==========
def extract_fvd_embeddings(
    dataset,
    batch_size=4,
    num_workers=8,
    device="cuda",
    ddp=False,
    gather_on_cpu=True,
    preflight=False,
    shuffle_seed=0,
    i3d_pool="avg",  # "avg" | "max" | "flat"
):
    """
    Extract I3D features for FVD.
    If get_fvd_logits returns [B, D, H, W], we spatially pool to [B, D].
    """
    from torch.nn import functional as F

    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False, seed=shuffle_seed)
        if ddp
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_skip_none,
    )
    i3d = load_fvd_model(device)
    kept = []
    did_preflight = False

    for bidx, batch in enumerate(
        tqdm(
            loader,
            desc="Extracting FVD Embeddings",
            disable=(dist.is_initialized() and dist.get_rank() != 0),
        )
    ):
        try:
            if not torch.is_tensor(batch):
                raise RuntimeError(f"Batch is not a tensor: {type(batch)}")
            batch = batch.contiguous()
            if batch.dtype != torch.uint8:
                batch = batch.to(torch.uint8)
            np_batch = batch.cpu().numpy()  # [B, T, H, W, C], uint8

            if preflight and not did_preflight:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                torch.cuda.synchronize()

            emb = get_fvd_logits(
                np_batch, i3d=i3d, device=device, batch_size=batch_size
            )
            # emb could be [B, D] or [B, D, H, W]
            if not torch.is_tensor(emb):
                raise RuntimeError(f"get_fvd_logits returned non-tensor: {type(emb)}")

            if emb.ndim == 4:
                # Pool spatial dims -> [B, D]
                if i3d_pool == "avg":
                    emb = emb.mean(dim=(2, 3))
                elif i3d_pool == "max":
                    emb = torch.amax(emb, dim=(2, 3))
                elif i3d_pool == "flat":
                    B, D, H, W = emb.shape
                    emb = emb.reshape(B, D * H * W)
                else:
                    raise ValueError(f"Unknown i3d_pool='{i3d_pool}'")
            elif emb.ndim != 2:
                raise RuntimeError(
                    f"get_fvd_logits returned unexpected shape: {tuple(emb.shape)}"
                )

            # Ensure contiguous fp32 on device
            emb = emb.to(
                device=device, dtype=torch.float32, non_blocking=True
            ).contiguous()
            kept.append(emb)

            if preflight and not did_preflight:
                torch.cuda.synchronize()
                did_preflight = True

        except Exception as e:
            ddp_print(f"[FVD] Skipping batch {bidx} due to error: {repr(e)}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    local = torch.cat(kept, dim=0) if kept else torch.empty(0, 400, device=device)

    if ddp and dist.is_initialized():
        if gather_on_cpu:
            cpu_local = local.detach().float().cpu()
            gathered = gather_cpu_objects(cpu_local)  # list of CPU tensors (per rank)
            all_cpu = concat_gathered_cpu_tensors(gathered)
            return all_cpu.to(device)
        else:
            return gather_embeddings_no_pad_mismatch(local, device=device)
    else:
        return local


# ========== FID (Inception-V3 avgpool) ==========
@torch.no_grad()
def build_inception_embedder(device: str = "cuda"):
    import torchvision
    from torchvision.models import Inception_V3_Weights
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = torchvision.models.inception_v3(weights=weights)
    model.eval().to(device)
    extractor = create_feature_extractor(
        model, return_nodes={"avgpool": "feat"}
    )  # -> [N, 2048, 1, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    return extractor, (mean, std)


@torch.no_grad()
def extract_fid_features(
    dataset,
    batch_size=8,
    num_workers=8,
    device="cuda",
    ddp=False,
    gather_on_cpu=True,
    shuffle_seed=0,
):
    import torch.nn.functional as F

    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False, seed=shuffle_seed)
        if ddp
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_skip_none,
    )
    extractor, (mean, std) = build_inception_embedder(device)
    feats = []
    for bidx, batch in enumerate(
        tqdm(
            loader,
            desc="Extracting FID Features",
            disable=(dist.is_initialized() and dist.get_rank() != 0),
        )
    ):
        try:
            B, T, H, W, C = batch.shape
            imgs = (
                batch.reshape(B * T, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            if imgs.shape[-1] != 299 or imgs.shape[-2] != 299:
                imgs = F.interpolate(
                    imgs, size=(299, 299), mode="bilinear", align_corners=False
                )
            imgs = (imgs - mean) / std
            out = extractor(imgs)["feat"].flatten(1)  # [N, 2048] on device
            feats.append(out)
        except Exception as e:
            ddp_print(f"[FID] Skipping batch {bidx} due to error: {repr(e)}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    local = torch.cat(feats, dim=0) if feats else torch.empty(0, 2048, device=device)
    if ddp and dist.is_initialized():
        if gather_on_cpu:
            cpu_local = local.detach().float().cpu()
            gathered = gather_cpu_objects(cpu_local)
            all_cpu = concat_gathered_cpu_tensors(gathered)
            return all_cpu.to(device)
        else:
            return gather_embeddings_no_pad_mismatch(local, device=device)
    else:
        return local


# ========== Robust Fréchet (NumPy/SciPy) ==========
def frechet_from_features_numpy(
    feats_gen: torch.Tensor, feats_real: torch.Tensor, eps: float = 1e-6
) -> float:
    """
    Compute Fréchet distance on CPU with SciPy, robustly.
    Inputs are torch [N, D] (any device). Returns Python float.
    """
    import scipy.linalg as linalg

    G = feats_gen.detach().float().cpu().numpy()
    R = feats_real.detach().float().cpu().numpy()
    if G.ndim != 2 or R.ndim != 2:
        raise ValueError(f"Features must be [N, D]. Got G:{G.shape}, R:{R.shape}")
    if G.shape[0] < 2 or R.shape[0] < 2:
        mu_g, mu_r = G.mean(axis=0), R.mean(axis=0)
        diff = mu_g - mu_r
        return float(np.dot(diff, diff))

    mu_g, mu_r = G.mean(axis=0), R.mean(axis=0)
    sigma_g = np.cov(G, rowvar=False).astype(np.float64)
    sigma_r = np.cov(R, rowvar=False).astype(np.float64)

    d = mu_g.shape[0]
    epsI = eps * np.eye(d, dtype=np.float64)
    sigma_g += epsI
    sigma_r += epsI

    diff = (mu_g - mu_r).astype(np.float64)

    covmean, _ = linalg.sqrtm(sigma_g @ sigma_r, disp=False)
    if not np.isfinite(covmean).all():
        covmean, _ = linalg.sqrtm(
            (sigma_g + 10 * epsI) @ (sigma_r + 10 * epsI), disp=False
        )
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff.dot(diff) + np.trace(sigma_g) + np.trace(sigma_r) - 2.0 * np.trace(covmean)
    )
    return float(fid)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(
        "FID & FVD for matched pairs (real from JSON; gens named by underscore captions)"
    )
    parser.add_argument(
        "--json_map",
        type=str,
        required=True,
        help="JSON array of objects with 'video_path' and 'caption'.",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Folder with generated videos whose filenames are caption_with_underscores.mp4",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", type=str, default="mp4,webm,mkv,avi,mov,m4v")
    parser.add_argument("--num_eval", type=int, default=1000000)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddp", action="store_true")

    # Matching controls (permissive defaults)
    parser.add_argument(
        "--lowercase_match",
        action="store_true",
        help="Lowercase stems for matching (recommended).",
    )
    parser.add_argument(
        "--fuzzy_cutoff",
        type=float,
        default=0.70,
        help="difflib cutoff (0..1). Lower catches more.",
    )
    parser.add_argument(
        "--min_prefix_chars",
        type=int,
        default=40,
        help="Minimum prefix length to accept for truncated names.",
    )
    parser.add_argument(
        "--min_jaccard",
        type=float,
        default=0.35,
        help="Minimum token Jaccard similarity to accept a match.",
    )

    # Robust DDP collection & debugging
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gather_on_cpu",
        dest="gather_on_cpu",
        action="store_true",
        help="Gather features across ranks on CPU via all_gather_object (safer).",
    )
    group.add_argument(
        "--gather_on_gpu",
        dest="gather_on_cpu",
        action="store_false",
        help="Use original GPU/NCCL gather.",
    )
    parser.set_defaults(gather_on_cpu=True)

    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Seed for DistributedSampler to get stable shard ordering.",
    )
    parser.add_argument(
        "--preflight_fvd",
        action="store_true",
        help="Run a single FVD batch with CUDA_LAUNCH_BLOCKING to catch early issues.",
    )

    parser.add_argument("--save_pairs", type=str, default="matched_pairs.tsv")
    parser.add_argument("--save_unmatched", type=str, default="unmatched.jsonl")
    parser.add_argument("--save_debug", type=str, default="matching_debug_samples.tsv")

    args = parser.parse_args()

    setup_ddp(args)
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    ddp_print(f"Using device: {device}")

    # Suggested env knobs (safe if already set)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    try:
        # ---- load JSON list ----
        if not os.path.isfile(args.json_map):
            raise FileNotFoundError(f"json_map not found: {args.json_map}")
        with open(args.json_map, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("json_map must be a JSON array of objects.")

        exts = [e.strip() for e in args.extensions.split(",") if e.strip()]
        gen_paths_all = list_videos(
            args.gen_dir, recursive=args.recursive, extensions=exts
        )
        if len(gen_paths_all) == 0:
            raise RuntimeError(f"No generated videos found in {args.gen_dir}")

        # ---- build matched pairs ----
        real_paths, gen_paths, unmatched, debug_pairs = match_pairs_from_json(
            data,
            gen_paths_all,
            lowercase_match=args.lowercase_match,
            fuzzy_cutoff=args.fuzzy_cutoff,
            min_prefix_chars=args.min_prefix_chars,
            min_jaccard=args.min_jaccard,
            limit=args.num_eval,
        )

        N = min(len(real_paths), len(gen_paths))
        real_paths, gen_paths = real_paths[:N], gen_paths[:N]
        ddp_print(
            f"Matched pairs: {N} | Generated files in dir: {len(gen_paths_all)} | Unmatched: {len(unmatched)}"
        )

        # Save pairs + unmatched + debug (rank 0)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            pairs_path = os.path.join(args.gen_dir, args.save_pairs)
            with open(pairs_path, "w") as f:
                for r, g in zip(real_paths, gen_paths):
                    f.write(f"{r}\t{g}\n")
            ddp_print(f"Saved matched pairs to: {pairs_path}")

            unmatched_path = os.path.join(args.gen_dir, args.save_unmatched)
            with open(unmatched_path, "w") as f:
                for u in unmatched:
                    f.write(json.dumps(u, ensure_ascii=False) + "\n")
            ddp_print(f"Saved unmatched list to: {unmatched_path}")

            dbg_path = os.path.join(args.gen_dir, args.save_debug)
            with open(dbg_path, "w") as f:
                f.write("want_key\tmatched_stem\n")
                for wk, mk in debug_pairs[:200]:
                    f.write(f"{wk}\t{mk}\n")
            ddp_print(f"Saved debug samples to: {dbg_path}")

        if N == 0:
            raise RuntimeError(
                "No matched pairs were found. Try lowering --fuzzy_cutoff, --min_prefix_chars, or --min_jaccard."
            )

        # ---- Datasets (paired, same order) ----
        target_size = (args.image_size, args.image_size)
        real_ds = MP4VideoDataset(
            real_paths, num_frames=args.num_frames, target_size=target_size
        )
        gen_ds = MP4VideoDataset(
            gen_paths, num_frames=args.num_frames, target_size=target_size
        )

        # ----- FID features -----
        real_fid_feats = extract_fid_features(
            real_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
            gather_on_cpu=args.gather_on_cpu,
            shuffle_seed=args.shuffle_seed,
        )
        gen_fid_feats = extract_fid_features(
            gen_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
            gather_on_cpu=args.gather_on_cpu,
            shuffle_seed=args.shuffle_seed,
        )

        # ----- FVD embeddings -----
        real_fvd = extract_fvd_embeddings(
            real_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
            gather_on_cpu=args.gather_on_cpu,
            preflight=args.preflight_fvd,
            shuffle_seed=args.shuffle_seed,
        )
        gen_fvd = extract_fvd_embeddings(
            gen_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
            gather_on_cpu=args.gather_on_cpu,
            preflight=args.preflight_fvd,
            shuffle_seed=args.shuffle_seed,
        )

        # ----- Fréchet distances -----
        fid_score = frechet_from_features_numpy(gen_fid_feats, real_fid_feats)
        fvd_score = frechet_from_features_numpy(gen_fvd, real_fvd)

        # ----- Save (rank 0) -----
        if not dist.is_initialized() or dist.get_rank() == 0:
            results_path = os.path.join(args.gen_dir, "results_matched.txt")
            with open(results_path, "w") as f:
                f.write(f"PAIRS: {N}\n")
                f.write(f"FID: {fid_score:.4f}\n")
                f.write(f"FVD: {fvd_score:.4f}\n")
                f.write(f"pairs_file: {os.path.join(args.gen_dir, args.save_pairs)}\n")
                f.write(
                    f"unmatched_file: {os.path.join(args.gen_dir, args.save_unmatched)}\n"
                )
                f.write(f"debug_file: {os.path.join(args.gen_dir, args.save_debug)}\n")
            print("\n====== Results (Matched) ======")
            print(f"PAIRS: {N}")
            print(f"FID: {fid_score:.4f}")
            print(f"FVD: {fvd_score:.4f}")
            print(f"Saved to: {results_path}")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
