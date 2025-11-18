#!/usr/bin/env python3
# generate_cogvideox_t2v_ddp.py
import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
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

# Diffusers schedulers
from diffusers import (
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)

# Videox-fun imports
from videox_fun.dist import shard_model
from videox_fun.models import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    T5EncoderModel,
    T5Tokenizer,
)
from videox_fun.pipeline import CogVideoXFunInpaintPipeline, CogVideoXFunPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "CogVideoX T2V DDP sampler (naming/sharding like your TI2V script)"
    )

    # dataset & IO
    p.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="JSON array with items; we only need a caption field (caption/caption_aug/caption_contra).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Where to write mp4s. Will create T2V/<CAPSRC>/[category]",
    )
    p.add_argument(
        "--resume", action="store_true", help="Skip items whose output already exists."
    )
    p.add_argument(
        "--prompt_category",
        type=str,
        default=None,
        help="Optional subfolder under the caption source (e.g., action_binding)",
    )

    # caption picking (same policy as TI2V)
    p.add_argument(
        "--caption_field",
        type=str,
        default="auto",
        choices=["auto", "base", "aug", "contra"],
        help="base->caption, aug->caption_aug, contra->caption_contra, auto tries aug->contra->base",
    )

    # model / paths
    p.add_argument(
        "--model_dir",
        type=str,
        default="models/CogVideoX1.5-5B",
        help="Path to CogVideoX model folder that contains subfolders: transformer, vae, tokenizer, text_encoder, scheduler",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="DDIM_Origin",
        choices=["Euler", "EulerA", "DPMpp", "PNDM", "DDIM_Cog", "DDIM_Origin"],
    )

    # memory / speed
    p.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="model_cpu_offload",
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
        help="Compile transformer blocks (not compatible with sequential_cpu_offload/FSDP).",
    )
    p.add_argument("--fsdp_dit", action="store_true", help="FSDP shard transformer")
    p.add_argument(
        "--fsdp_text_encoder", action="store_true", help="FSDP shard text encoder"
    )

    # generation
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=672)
    p.add_argument(
        "--num_frames", type=int, default=49, help="V1.0/1.1: ≤49, V1.5: ≤85"
    )
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument(
        "--ref_field",
        type=str,
        default=None,
        help="JSON key that contains per-item reference images (e.g., 'file_name').",
    )
    p.add_argument(
        "--ref_frame",
        type=str,
        default=None,
        help="Optional path to a reference image. When omitted, generation stays pure T2V.",
    )
    p.add_argument(
        "--ti2v_mode",
        type=str,
        default="start",
        choices=["start", "mid", "last", "random"],
        help="Placement of the reference image when --ref_frame is provided.",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="overexposed, blurry, low quality, watermark, solid background, distorted body/trajectory",
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument(
        "--weight_dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--text_encoder_dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--vae_dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )

    # LoRA (optional)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_weight", type=float, default=0.55)

    # batching / sharding
    p.add_argument(
        "--max_videos", type=int, default=None, help="Debug: limit total items."
    )
    p.add_argument("--verbose", action="store_true")

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
# Caption helpers (same policy as TI2V)
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


def resolve_module_device(module, fallback_device: torch.device) -> torch.device:
    """
    Try to infer the real device of a module, accounting for potential offload/meta devices.
    """

    def _to_device(obj):
        if obj is None:
            return None
        try:
            return torch.device(obj)
        except (TypeError, ValueError):
            return None

    dev = _to_device(getattr(module, "device", None))
    if dev is not None and dev.type != "meta":
        return dev

    try:
        param_dev = next(module.parameters()).device  # type: ignore[attr-defined]
        dev = _to_device(param_dev)
        if dev is not None and dev.type != "meta":
            return dev
    except (StopIteration, AttributeError):
        pass

    return _to_device(fallback_device) or torch.device("cpu")


def load_ref_image_as_video(
    path: str, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return tensor shaped [1,1,3,H,W] with pixel values in [0,1]."""
    img = Image.open(path).convert("RGB")
    try:
        resample = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:
        resample = Image.BICUBIC
    img = img.resize((width, height), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous()
    )
    target_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    return tensor.to(device=device, dtype=target_dtype, non_blocking=True)


# ---------------------------
# Scheduler mapping
# ---------------------------
def build_scheduler(name: str, model_dir: str):
    if name == "Euler":
        S = EulerDiscreteScheduler
    elif name == "EulerA":
        S = EulerAncestralDiscreteScheduler
    elif name == "DPMpp":
        S = DPMSolverMultistepScheduler
    elif name == "PNDM":
        S = PNDMScheduler
    elif name == "DDIM_Cog":
        S = CogVideoXDDIMScheduler
    else:
        S = DDIMScheduler  # "DDIM_Origin"
    return S.from_pretrained(model_dir, subfolder="scheduler")


# ---------------------------
# Model loading
# ---------------------------
def load_models(args, device):
    weight_dtype = getattr(torch, args.weight_dtype)
    text_encoder_dtype = getattr(torch, args.text_encoder_dtype)
    vae_dtype = getattr(torch, args.vae_dtype)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_dir, subfolder="vae").to(
        vae_dtype
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_dir, subfolder="text_encoder", torch_dtype=text_encoder_dtype
    )

    scheduler = build_scheduler(args.sampler, args.model_dir)

    # Inpaint vs. base pipeline
    if transformer.config.in_channels != vae.config.latent_channels:
        pipe = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipe = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

    # Optional compile
    if args.compile_dit:
        for i in range(len(pipe.transformer.transformer_blocks)):
            pipe.transformer.transformer_blocks[i] = torch.compile(
                pipe.transformer.transformer_blocks[i]
            )

    # Memory/offload modes
    if args.gpu_memory_mode == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(
            transformer, exclude_module_name=[], device=device
        )
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipe.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(
            transformer, exclude_module_name=[], device=device
        )
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipe.to(device=device)
    else:  # model_full_load
        pipe.to(device=device)

    # Optional FSDP sharding (same switches you use)
    if (ulysses := int(os.environ.get("ULYSSES_DEGREE", "1"))) > 1 or (
        ring := int(os.environ.get("RING_DEGREE", "1"))
    ) > 1:
        pipe.transformer.enable_multi_gpus_inference()
    if args.fsdp_dit:
        from functools import partial

        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipe.transformer = shard_fn(pipe.transformer)
    if args.fsdp_text_encoder:
        from functools import partial

        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipe.text_encoder = shard_fn(pipe.text_encoder)

    return pipe, transformer, vae


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

    # Keep only items that have a usable caption
    items = []
    for i, it in enumerate(data):
        cap, tag = pick_caption(it, args.caption_field)
        if not cap:
            continue
        ref_value = ""
        if args.ref_field:
            ref_value = (it.get(args.ref_field) or "").strip()
            if not ref_value:
                continue
        items.append(
            {
                "id": i,
                "caption": cap,
                "cap_tag": tag,
                "orig_cap": (it.get("original_caption") or "").strip(),
                "ref_value": ref_value,
            }
        )  # BASE / AUG / CONTRA

    if args.max_videos is not None:
        items = items[: args.max_videos]

    # Shard by rank
    sharded = [it for i, it in enumerate(items) if i % world_size == rank]
    _log(f"[Shard] rank {rank} got {len(sharded)} / {len(items)} items")

    # Prepare output dir (T2V/TI2V/<CAPSRC>/[category])
    ti2v_requested = bool(args.ref_frame or args.ref_field)
    mode_dir = "TI2V" if ti2v_requested else "T2V"
    cap_tag_for_dir = sharded[0]["cap_tag"] if sharded else "BASE"
    if args.prompt_category is not None:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir / args.prompt_category
    else:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Load models
    pipe, transformer, vae = load_models(args, device=device)

    weight_dtype = getattr(torch, args.weight_dtype)
    text_encoder_dtype = getattr(torch, args.text_encoder_dtype)
    vae_dtype = getattr(torch, args.vae_dtype)

    lora_merged = False
    if args.lora_path:
        pipe = merge_lora(
            pipe,
            args.lora_path,
            args.lora_weight,
            device=device,
            dtype=weight_dtype,
        )
        lora_merged = True

    supports_ti2v = transformer.config.in_channels == vae.config.latent_channels
    if ti2v_requested and not supports_ti2v:
        raise ValueError(
            "TI2V references (--ref_frame/--ref_field) require the base pipeline where transformer and VAE latent channels match."
        )

    vae_device = resolve_module_device(vae, device)

    if supports_ti2v:

        def _encode_conditioning_pixels_on_cpu(cond_pixels_lat: torch.Tensor):
            cond_cpu = cond_pixels_lat.to("cpu", dtype=torch.float32, non_blocking=True)
            vae_cpu = AutoencoderKLCogVideoX.from_pretrained(
                args.model_dir, subfolder="vae"
            ).to("cpu", dtype=torch.float32)
            with torch.autocast(device_type="cpu", enabled=False):
                lat = vae_cpu.encode(cond_cpu)[0].sample()
            sf = float(getattr(vae_cpu.config, "scaling_factor", 1.0))
            lat = lat * sf
            del vae_cpu
            gc.collect()

            target_device = resolve_module_device(pipe.vae, device)
            target_dtype = getattr(pipe.vae, "dtype", lat.dtype)
            return lat.to(device=target_device, dtype=target_dtype, non_blocking=True)

        pipe._encode_conditioning_pixels = _encode_conditioning_pixels_on_cpu

    static_ref_video = None
    if args.ref_frame and supports_ti2v:
        ref_path = os.path.abspath(os.path.expanduser(args.ref_frame.strip()))
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        static_ref_video = load_ref_image_as_video(
            ref_path, args.height, args.width, vae_device, vae_dtype
        )
        log(
            rank,
            f"[TI2V] Loaded reference frame {ref_path} with mode '{args.ti2v_mode}'.",
        )

    # dtype + generator
    base_gen = torch.Generator(device=device)

    # Adjust temporal length to VAE stride and transformer multiple
    T = args.num_frames
    tcr = vae.config.temporal_compression_ratio
    if T != 1:
        T = int((T - 1) // tcr * tcr + 1)

    latent_frames = (T - 1) // tcr + 1 if T != 1 else 1
    if T != 1 and getattr(transformer.config, "patch_size_t", None) is not None:
        p = transformer.config.patch_size_t
        if latent_frames % p != 0:
            add_lat = p - (latent_frames % p)
            T += add_lat * tcr
            latent_frames += add_lat

    # Process shard
    for it in sharded:
        prompt = it["caption"]
        cap_tag = it["cap_tag"]  # BASE / AUG / CONTRA

        # Recompute out_root per item (keeps folder stable if tags differ in same shard)
        if args.prompt_category is not None:
            out_root = Path(args.outdir) / mode_dir / cap_tag / args.prompt_category
        else:
            out_root = Path(args.outdir) / mode_dir / cap_tag
        out_root.mkdir(parents=True, exist_ok=True)

        # Filename: slugified prompt (same as TI2V)
        # prompt_stem = slugify(prompt) if prompt else f"sample_{it['id']:05d}"
        # prompt_stem = prompt_stem[:180] if len(prompt_stem) > 180 else prompt_stem
        save_caption = it.get("orig_cap")
        save_stem = save_caption if save_caption else prompt
        save_stem = save_stem[:180] if len(save_stem) > 180 else save_stem
        base_name = save_stem if save_stem else f"sample_{it['id']:05d}"
        name_with_rank = f"{base_name}_idx{it['id']:05d}_r{rank}"
        out_path = out_root / f"{name_with_rank}.mp4"

        if args.resume and out_path.exists():
            _log(f"[Skip] exists: {out_path}")
            continue
        if out_path.exists():
            # Append _1, _2, ...
            i = 1
            while True:
                cand = out_root / f"{name_with_rank}_{i}.mp4"
                if not cand.exists():
                    out_path = cand
                    break
                i += 1

        gen = base_gen.manual_seed(args.seed + it["id"])
        with torch.no_grad():
            if transformer.config.in_channels != vae.config.latent_channels:
                # Inpaint variant expects video_length inside the pipeline;
                # pass dummy latents & mask with the **correct frames + size**.
                input_video, input_video_mask, _ = get_image_to_video_latent(
                    None, None, video_length=T, sample_size=[args.height, args.width]
                )

                sample = pipe(
                    prompt,
                    num_frames=T,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    generator=gen,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    video=input_video,
                    mask_video=input_video_mask,
                ).videos
            else:
                item_ref_video = None
                if supports_ti2v:
                    if static_ref_video is not None:
                        item_ref_video = static_ref_video
                    elif args.ref_field:
                        raw_ref = (it.get("ref_value") or "").strip()
                        resolved_ref = os.path.abspath(os.path.expanduser(raw_ref))
                        if not resolved_ref or not os.path.exists(resolved_ref):
                            log(
                                rank,
                                f"[Skip] reference image missing for item {it['id']}: {resolved_ref or '(empty)'}",
                            )
                            continue
                        item_ref_video = load_ref_image_as_video(
                            resolved_ref, args.height, args.width, vae_device, vae_dtype
                        )

                pipe_kwargs = dict(
                    prompt=prompt,
                    num_frames=T,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    generator=gen,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                )
                if item_ref_video is not None:
                    pipe_kwargs["video"] = item_ref_video
                    pipe_kwargs["ti2v_mode"] = args.ti2v_mode
                sample = pipe(**pipe_kwargs).videos

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(sample, str(out_path), fps=args.fps)
        _log(f"[OK] {it['id']:05d} -> {out_path}")

    if lora_merged:
        pipe = unmerge_lora(
            pipe,
            args.lora_path,
            args.lora_weight,
            device=device,
            dtype=weight_dtype,
        )

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
