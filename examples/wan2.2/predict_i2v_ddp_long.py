#!/usr/bin/env python3
# generate_i2v_ti2v_ddp.py
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image

# ---- project roots (same pattern you use) ----
this = os.path.abspath(__file__)
for r in [
    os.path.dirname(this),
    os.path.dirname(os.path.dirname(this)),
    os.path.dirname(os.path.dirname(os.path.dirname(this))),
]:
    if r not in sys.path:
        sys.path.insert(0, r)

# Wan 2.2 imports from your codebase
from diffusers import FlowMatchEulerDiscreteScheduler

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    get_image_to_video_latent_simple,
    save_videos_grid,
)


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Wan2.2 I2V/TI2V DDP sampler (same naming/sharding as your TI2V script)"
    )

    # dataset & IO
    p.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="JSON array with items having image refs ('image_path' or 'file_name'). For TI2V, also needs captions.",
    )
    p.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Optional root for relative image paths; ignored when JSON has absolute image_path.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Where to write mp4s. Will create I2V/TI2V/<CAPSRC>/[category] (CAPSRC used only for TI2V).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="i2v",
        choices=["i2v", "ti2v"],
        help="i2v=image only, ti2v=image+text",
    )
    p.add_argument(
        "--resume", action="store_true", help="Skip items whose output already exists."
    )
    p.add_argument(
        "--prompt_category",
        type=str,
        default=None,
        help="Optional subfolder (e.g., action_binding)",
    )

    # caption picking (same policy as your TI2V script)
    p.add_argument(
        "--caption_field",
        type=str,
        default="auto",
        choices=["auto", "base", "aug", "contra"],
        help="base->caption, aug->caption_aug, contra->caption_contra, auto tries aug->contra->base",
    )

    # model / config
    p.add_argument(
        "--config_path", type=str, default="config/wan2.2/wan_civitai_i2v.yaml"
    )
    p.add_argument("--model_dir", type=str, default="models/Wan2.2-I2V-A14B")

    p.add_argument(
        "--sampler",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    p.add_argument(
        "--shift",
        type=int,
        default=5,
        help="Noise schedule shift for Flow_Unipc / Flow_DPM++",
    )

    p.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="sequential_cpu_offload",
        choices=[
            "model_full_load",
            "model_full_load_and_qfloat8",
            "model_cpu_offload",
            "model_cpu_offload_and_qfloat8",
            "sequential_cpu_offload",
        ],
    )
    p.add_argument(
        "--compile_dit",
        action="store_true",
        help="Compile blocks (not compatible with sequential_cpu_offload/FSDP).",
    )

    # generation
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="overexposed, blurry, low quality, deformed hands, ugly, artifacts, static scene",
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument(
        "--weight_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"]
    )

    # TeaCache / cfg-skip / riflex
    p.add_argument("--enable_teacache", action="store_true", default=True)
    p.add_argument("--teacache_threshold", type=float, default=0.10)
    p.add_argument("--teacache_skip_start", type=int, default=5)
    p.add_argument("--teacache_offload", action="store_true", default=False)
    p.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    p.add_argument("--enable_riflex", action="store_true", default=False)
    p.add_argument("--riflex_k", type=int, default=6)

    # LoRA (optional)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_high_path", type=str, default=None)
    p.add_argument("--lora_weight", type=float, default=0.55)
    p.add_argument("--lora_high_weight", type=float, default=0.55)

    # batching / sharding
    p.add_argument(
        "--max_videos", type=int, default=None, help="Debug: limit total items."
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--positions_start",
        type=int,
        nargs="+",
        default=None,
        help="e.g., --positions_start -1 3 5",
    )

    return p.parse_args()


# ---------------------------
# DDP utils
# ---------------------------
def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def log(rank, *msg):
    if rank == 0:
        print(*msg, flush=True)


# ---------------------------
# Helpers (same as TI2V)
# ---------------------------
_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = _slug_re.sub("_", s)
    return s[:200]


def pick_caption(it, caption_field: str) -> tuple[str, str]:
    """
    Returns (caption_text, source_tag) where source_tag in {'BASE','AUG','CONTRA'}.
    If caption_field='auto', tries caption_aug -> caption_contra -> caption.
    """
    if caption_field == "base":
        cap = (it.get("caption") or "").strip()
        return cap, "BASE"
    if caption_field == "aug":
        cap = (it.get("caption_aug") or "").strip()
        return cap, "AUG"
    if caption_field == "contra":
        cap = (it.get("caption_contra") or "").strip()
        return cap, "CONTRA"
    # auto
    for key, tag in (
        ("caption_aug", "AUG"),
        ("caption_contra", "CONTRA"),
        ("caption", "BASE"),
    ):
        cap = (it.get(key) or "").strip()
        if cap:
            return cap, tag
    return "", "BASE"


def build_scheduler(name, cfg, shift):
    if name == "Flow":
        S = FlowMatchEulerDiscreteScheduler
    elif name == "Flow_Unipc":
        S = FlowUniPCMultistepScheduler
        cfg["shift"] = 1
    else:
        S = FlowDPMSolverMultistepScheduler
        cfg["shift"] = 1
    return S(**filter_kwargs(S, cfg))


def resolve_img_path(item, images_root: str | None) -> str:
    """
    Prefer absolute item['image_path'] if present.
    Else use item['file_name'] joined with images_root (if provided).
    Returns '' if we can't resolve a path.
    """
    p = (item.get("image_path") or "").strip()
    if p:
        return p if os.path.isabs(p) else os.path.join(images_root or "", p)
    fn = (item.get("file_name") or "").strip()
    if not fn:
        return ""
    return fn if os.path.isabs(fn) else os.path.join(images_root or "", fn)


def make_unique_path(base_path: Path) -> Path:
    """Append _1, _2, ... if the file already exists."""
    if not base_path.exists():
        return base_path
    stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


# ---------------------------
# Model loading
# ---------------------------
def load_models(args, device):
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_dir,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, args.weight_dtype),
    )

    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_dir,
            config["transformer_additional_kwargs"].get(
                "transformer_high_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, args.weight_dtype),
    )

    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_dir, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(getattr(torch, args.weight_dtype))

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model_dir,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model_dir,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, args.weight_dtype),
    )

    scheduler = build_scheduler(
        args.sampler, OmegaConf.to_container(config["scheduler_kwargs"]), args.shift
    )

    pipe = Wan2_2I2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Optional compile
    if args.compile_dit:
        for i in range(len(pipe.transformer.blocks)):
            pipe.transformer.blocks[i] = torch.compile(pipe.transformer.blocks[i])
        for i in range(len(pipe.transformer_2.blocks)):
            pipe.transformer_2.blocks[i] = torch.compile(pipe.transformer_2.blocks[i])

    # Memory/offload modes
    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(pipe.transformer, ["modulation"], device=device)
        replace_parameters_by_name(pipe.transformer_2, ["modulation"], device=device)
        pipe.transformer.freqs = pipe.transformer.freqs.to(device=device)
        pipe.transformer_2.freqs = pipe.transformer_2.freqs.to(device=device)
        pipe.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(
            pipe.transformer, exclude_module_name=["modulation"], device=device
        )
        convert_model_weight_to_float8(
            pipe.transformer_2, exclude_module_name=["modulation"], device=device
        )
        convert_weight_dtype_wrapper(
            pipe.transformer, getattr(torch, args.weight_dtype)
        )
        convert_weight_dtype_wrapper(
            pipe.transformer_2, getattr(torch, args.weight_dtype)
        )
        pipe.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(
            pipe.transformer, exclude_module_name=["modulation"], device=device
        )
        convert_model_weight_to_float8(
            pipe.transformer_2, exclude_module_name=["modulation"], device=device
        )
        convert_weight_dtype_wrapper(
            pipe.transformer, getattr(torch, args.weight_dtype)
        )
        convert_weight_dtype_wrapper(
            pipe.transformer_2, getattr(torch, args.weight_dtype)
        )
        pipe.to(device=device)
    else:
        pipe.to(device=device)

    # TeaCache / cfg-skip
    if args.enable_teacache:
        coeffs = get_teacache_coefficients(args.model_dir)
        if coeffs is not None:
            pipe.transformer.enable_teacache(
                coeffs,
                args.steps,
                args.teacache_threshold,
                num_skip_start_steps=args.teacache_skip_start,
                offload=args.teacache_offload,
            )
            pipe.transformer_2.share_teacache(transformer=pipe.transformer)

    if args.cfg_skip_ratio is not None and args.cfg_skip_ratio > 0:
        pipe.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.steps)
        pipe.transformer_2.share_cfg_skip(transformer=pipe.transformer)

    return pipe, config


def ensure_latent_video(img_path, T, H, W, positions=None):
    """
    Prepare latent video with image conditioning at specified positions.

    Args:
        img_path: Path to image or list of image paths
        T: Number of frames
        H: Height
        W: Width
        positions: List of frame indices (defaults to [0] if None)
    """
    if img_path is None:
        return None, None, None
    if positions is None:
        print("Using original conditioning at frame 0")
        return get_image_to_video_latent(
            img_path, None, video_length=T, sample_size=[H, W]
        )
    else:
        print(f"Using new conditioning at frame {positions[0]}")
        return get_image_to_video_latent_simple(
            images=img_path, positions=positions, video_length=T, sample_size=[H, W]
        )


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    rank, world_size, local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    def _log(*m):
        if rank == 0 and args.verbose:
            print(*m, flush=True)

    # Load dataset
    with open(args.dataset_json, "r") as f:
        data = json.load(f)

    # Normalize & filter items
    items = []
    for i, it in enumerate(data):
        # Require an image reference
        has_img = bool((it.get("image_path") or it.get("file_name") or "").strip())
        if not has_img:
            continue

        cap, tag = pick_caption(it, args.caption_field)
        if args.mode == "ti2v" and not cap:
            # TI2V needs caption + image
            continue

        items.append(
            {
                "id": i,
                "file_name": (it.get("file_name") or "").strip(),
                "image_path": (it.get("image_path") or "").strip(),
                "caption": cap,  # may be empty in i2v
                "cap_tag": tag,  # BASE / AUG / CONTRA
                "orig_cap": (it.get("original_caption") or "").strip(),
            }
        )

    if args.max_videos is not None:
        items = items[: args.max_videos]

    # Shard by rank
    sharded = [it for i, it in enumerate(items) if i % world_size == rank]
    _log(f"[Shard] rank {rank} got {len(sharded)} / {len(items)} items")

    # Make base output dir
    mode_dir = dict(i2v="I2V", ti2v="TI2V")[args.mode]
    # If TI2V, folder includes caption source; else use "BASE" by convention
    cap_tag_for_dir = (
        sharded[0]["cap_tag"] if (sharded and args.mode == "ti2v") else "BASE"
    )
    if args.prompt_category is not None:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir / args.prompt_category
    else:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Load models / pipeline
    pipe, config = load_models(args, device=device)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    # dtype + generator
    weight_dtype = getattr(torch, args.weight_dtype)
    base_gen = torch.Generator(device=device)

    # Adjust temporal length to VAE stride
    T = args.num_frames
    tcr = pipe.vae.config.temporal_compression_ratio
    if T != 1:
        T = int((T - 1) // tcr * tcr + 1)

    if args.enable_riflex:
        latent_frames = (T - 1) // tcr + 1
        pipe.transformer.enable_riflex(k=args.riflex_k, L_test=latent_frames)
        pipe.transformer_2.enable_riflex(k=args.riflex_k, L_test=latent_frames)

    # Process shard
    for it in sharded:
        prompt = it["caption"] if args.mode == "ti2v" else ""
        cap_tag = it["cap_tag"] if args.mode == "ti2v" else "BASE"

        # Resolve image path
        img_path = resolve_img_path(it, args.images_root)
        if not img_path or not os.path.exists(img_path):
            _log(f"[Warn] image missing: {img_path or '(empty)'} (skip)")
            continue

        # Recompute out_root per item to reflect possibly different cap_tag in TI2V
        if args.prompt_category is not None:
            out_root = Path(args.outdir) / mode_dir / cap_tag / args.prompt_category
        else:
            out_root = Path(args.outdir) / mode_dir / cap_tag
        out_root.mkdir(parents=True, exist_ok=True)

        # Filename: prefer prompt slug; else fall back to image/file stem; else id
        fallback_stem = Path(img_path).stem if img_path else f"sample_{it['id']:05d}"
        # prompt_stem = slugify(prompt) if prompt else slugify(fallback_stem)
        prompt_stem = prompt[:180] if len(prompt) > 180 else prompt
        out_path = out_root / f"{prompt_stem}.mp4"
        # save_caption = it.get("orig_cap")
        # save_stem = save_caption if save_caption else prompt
        # out_path = out_root / f"{save_stem}.mp4"

        if args.resume and out_path.exists():
            _log(f"[Skip] exists: {out_path}")
            continue
        if out_path.exists():
            out_path = make_unique_path(out_path)

        # Prepare latent video + mask from image
        # v_in, m_in, _ = get_image_to_video_latent(
        #     img_path, None, video_length=T, sample_size=[args.height, args.width]
        # )
        input_video, input_video_mask, _ = ensure_latent_video(
            img_path,
            T=T,
            H=args.height,
            W=args.width,
            positions=args.positions_start,
        )

        gen = base_gen.manual_seed(args.seed + it["id"])
        with torch.no_grad():
            sample = pipe(
                prompt,
                num_frames=T,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                generator=gen,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                boundary=boundary,
                video=input_video,
                mask_video=input_video_mask,
                shift=args.shift,
            ).videos

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(sample, str(out_path), fps=args.fps)
        _log(f"[OK] {it['id']:05d} -> {out_path}")

    # Finish
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("[Done] All ranks finished.", flush=True)
        dist.destroy_process_group()
    else:
        print("[Done] Single process finished.", flush=True)


if __name__ == "__main__":
    main()
