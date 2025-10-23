#!/usr/bin/env python3
# evaluation/evaluate_quality.py
import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
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


def gather_embeddings_no_pad_mismatch(local_embs: torch.Tensor, device: str = "cuda"):
    """DDP-safe gather for variable-sized chunks; returns [sum(n_i), D]."""
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

        # Always produce exactly self.num_frames indices (uniform with repetition)
        n = len(frames)
        idxs = np.clip(
            np.round(np.linspace(0, n - 1, self.num_frames)).astype(int), 0, n - 1
        )
        selected = [frames[i] for i in idxs]

        # Resize and ensure contiguous
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
        if b is None:  # filter
            continue
        if isinstance(b, np.ndarray):
            b = np.ascontiguousarray(b)
            items.append(torch.from_numpy(b.copy()))  # fresh storage
        elif torch.is_tensor(b):
            items.append(b.contiguous().clone())
        else:
            raise TypeError(f"Unexpected batch element type: {type(b)}")
    if not items:
        raise ValueError("Empty batch after filtering out None samples.")
    return torch.stack(items, dim=0)  # [B, T, H, W, C] on CPU


# ========== FVD (I3D) ==========
def extract_fvd_embeddings(
    dataset, batch_size=4, num_workers=8, device="cuda", ddp=False
):
    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False) if ddp else None
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
    all_embeddings = []
    for batch in tqdm(
        loader,
        desc="Extracting FVD Embeddings",
        disable=(dist.is_initialized() and dist.get_rank() != 0),
    ):
        # batch: [B, T, H, W, C] uint8 on CPU
        np_batch = batch.cpu().numpy()
        emb = get_fvd_logits(
            np_batch, i3d=i3d, device=device, batch_size=batch_size
        )  # torch.Tensor [B, D]
        all_embeddings.append(emb)
    local = (
        torch.cat(all_embeddings, dim=0)
        if all_embeddings
        else torch.empty(0, 400, device=device)
    )
    if ddp and dist.is_initialized():
        emb_cat = gather_embeddings_no_pad_mismatch(local, device=device)
    else:
        emb_cat = local
    return emb_cat  # torch [N, D] on device


# ========== FID (Inception-V3 avgpool) ==========
@torch.no_grad()
def build_inception_embedder(device: str = "cuda"):
    import torchvision
    from torchvision.models import Inception_V3_Weights
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = torchvision.models.inception_v3(
        weights=weights
    )  # don't force aux_logits=False
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
    dataset, batch_size=8, num_workers=8, device="cuda", ddp=False
):
    import torch.nn.functional as F

    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False) if ddp else None
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
    for batch in tqdm(
        loader,
        desc="Extracting FID Features",
        disable=(dist.is_initialized() and dist.get_rank() != 0),
    ):
        # batch: [B, T, H, W, C] uint8 -> float in [0,1]
        B, T, H, W, C = batch.shape
        imgs = (
            batch.reshape(B * T, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        # Inception expects 299x299
        if imgs.shape[-1] != 299 or imgs.shape[-2] != 299:
            imgs = F.interpolate(
                imgs, size=(299, 299), mode="bilinear", align_corners=False
            )
        imgs = (imgs - mean) / std
        out = extractor(imgs)["feat"].flatten(1)  # [N, 2048] on device
        feats.append(out)
    local = torch.cat(feats, dim=0) if feats else torch.empty(0, 2048, device=device)
    if ddp and dist.is_initialized():
        all_feats = gather_embeddings_no_pad_mismatch(local, device=device)
    else:
        all_feats = local
    return all_feats  # torch [N, 2048] on device


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
        # Covariance is ill-defined with <2 samples; add jittered identity with mean-only distance
        mu_g, mu_r = G.mean(axis=0), R.mean(axis=0)
        diff = mu_g - mu_r
        return float(np.dot(diff, diff))

    mu_g, mu_r = G.mean(axis=0), R.mean(axis=0)  # [D]
    sigma_g = np.cov(G, rowvar=False)  # [D, D]
    sigma_r = np.cov(R, rowvar=False)  # [D, D]

    # Ensure 2D
    if sigma_g.ndim == 0:
        sigma_g = np.array([[sigma_g]])
    if sigma_r.ndim == 0:
        sigma_r = np.array([[sigma_r]])

    # Numerical stability
    d = mu_g.shape[0]
    epsI = eps * np.eye(d, dtype=np.float64)
    sigma_g = sigma_g.astype(np.float64) + epsI
    sigma_r = sigma_r.astype(np.float64) + epsI

    diff = (mu_g - mu_r).astype(np.float64)  # [D]

    # sqrtm of product
    covmean, _ = linalg.sqrtm(sigma_g @ sigma_r, disp=False)
    if not np.isfinite(covmean).all():
        # Fallback: more jitter
        covmean, _ = linalg.sqrtm(
            (sigma_g + 10 * epsI) @ (sigma_r + 10 * epsI), disp=False
        )

    # Take real part if small complex residuals due to numerics
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff.dot(diff) + np.trace(sigma_g) + np.trace(sigma_r) - 2.0 * np.trace(covmean)
    )
    return float(fid)


# ========== Matching utility (optional) ==========
def match_by_basename(reals: Sequence[str], gens: Sequence[str]):
    m_real = {Path(p).stem: p for p in reals}
    m_gen = {Path(p).stem: p for p in gens}
    common = sorted(set(m_real.keys()) & set(m_gen.keys()))
    return [m_real[k] for k in common], [m_gen[k] for k in common]


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(
        "Generic FID & FVD for folders of videos (real vs generated)"
    )
    parser.add_argument(
        "--real_dir", type=str, required=True, help="Folder with real/reference videos"
    )
    parser.add_argument(
        "--gen_dir", type=str, required=True, help="Folder with generated videos"
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", type=str, default="mp4,webm,mkv,avi,mov,m4v")
    parser.add_argument("--num_eval", type=int, default=5000)
    parser.add_argument("--num_frames", type=int, default=97)  # you used 97
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument(
        "--match_by", type=str, default="none", choices=["none", "basename"]
    )
    args = parser.parse_args()

    setup_ddp(args)
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    ddp_print(f"Using device: {device}")

    try:
        exts = [e.strip() for e in args.extensions.split(",") if e.strip()]
        real_paths = list_videos(
            args.real_dir, recursive=args.recursive, extensions=exts
        )
        gen_paths = list_videos(args.gen_dir, recursive=args.recursive, extensions=exts)

        if args.match_by == "basename":
            real_paths, gen_paths = match_by_basename(real_paths, gen_paths)
            ddp_print(f"Matched by basename: {len(real_paths)} pairs")

        if len(real_paths) == 0 or len(gen_paths) == 0:
            raise RuntimeError(
                f"No videos found. real={len(real_paths)} gen={len(gen_paths)}"
            )

        real_paths = real_paths[: args.num_eval]
        gen_paths = gen_paths[: args.num_eval]
        N = min(len(real_paths), len(gen_paths))
        real_paths, gen_paths = real_paths[:N], gen_paths[:N]
        ddp_print(f"Using N={N} videos from each set.")

        target_size = (args.image_size, args.image_size)
        real_ds = MP4VideoDataset(
            real_paths, num_frames=args.num_frames, target_size=target_size
        )
        gen_ds = MP4VideoDataset(
            gen_paths, num_frames=args.num_frames, target_size=target_size
        )

        # ----- FID features (torch) -----
        real_fid_feats = extract_fid_features(
            real_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
        )
        gen_fid_feats = extract_fid_features(
            gen_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
        )

        # ----- FVD embeddings (torch) -----
        real_fvd = extract_fvd_embeddings(
            real_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
        )
        gen_fvd = extract_fvd_embeddings(
            gen_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ddp=args.ddp,
        )

        # ----- Fréchet distances (robust NumPy) -----
        fid_score = frechet_from_features_numpy(gen_fid_feats, real_fid_feats)
        fvd_score = frechet_from_features_numpy(gen_fvd, real_fvd)

        # ----- Save (rank 0) -----
        if not dist.is_initialized() or dist.get_rank() == 0:
            results_path = os.path.join(args.gen_dir, "results.txt")
            with open(results_path, "w") as f:
                f.write(f"FID: {fid_score:.4f}\n")
                f.write(f"FVD: {fvd_score:.4f}\n")
            print("\n====== Results ======")
            print(f"FID: {fid_score:.4f}")
            print(f"FVD: {fvd_score:.4f}")
            print(f"Saved to: {results_path}")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

# torchrun --nnodes=1 --nproc_per_node=4 evaluation/evaluate_quality.py   --real_dir /capstor/store/cscs/swissai/a144/datasets/ssv2   --gen_dir  /capstor/scratch/cscs/mhasan/VideoX-Fun/samples/intraguide_0.2_1100   --dataset_name SSV2   --num_eval 5000   --num_frames 33   --image_size 256   --batch_size 10 --ddp
# pip install torchmetrics[image]
# torchrun --nnodes=1 --nproc_per_node=4 evaluation/evaluate_quality.py   --real_dir /capstor/store/cscs/swissai/a144/datasets/ssv2   --gen_dir  /capstor/scratch/cscs/mhasan/VideoX-Fun/samples/intraguide_0.2_1100   --dataset_name SSV2   --num_eval 5000   --num_frames 33   --image_size 256   --batch_size 10 --ddp
# pip install torchmetrics[image]


# torchrun --nnodes=1 --nproc_per_node=4 evaluation/evaluate_quality.py   --real_dir /capstor/store/cscs/swissai/a144/datasets/OpenVid-1M/validation   --gen_dir  samples/wan-videos-openvid/T2V/BASE  --num_eval 10000   --num_frames 121   --image_size 256   --batch_size 10 --ddp
