#!/usr/bin/env python3
"""
Wan2.2 TI2V DDP sampler (multi-transformer, TI2V conditioning like predict_t2v_inp.py).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

this = os.path.abspath(__file__)
for r in [
    os.path.dirname(this),
    os.path.dirname(os.path.dirname(this)),
    os.path.dirname(os.path.dirname(os.path.dirname(this))),
]:
    if r not in sys.path:
        sys.path.insert(0, r)

from videox_fun.dist import shard_model
from videox_fun.models import (
    AutoTokenizer,
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2Pipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, save_videos_grid


def parse_args():
    p = argparse.ArgumentParser("Wan2.2 TI2V DDP sampler")
    # dataset / IO
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prompt_category", type=str, default=None)
    p.add_argument(
        "--caption_field",
        type=str,
        default="auto",
        choices=["auto", "base", "aug", "contra"],
    )
    p.add_argument("--ref_field", type=str, default=None)
    p.add_argument("--ref_frame", type=str, default=None)
    p.add_argument(
        "--ti2v_mode",
        type=str,
        default="start",
        choices=["start", "mid", "last", "random"],
    )

    # model / config
    p.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.2/wan_civitai_t2v.yaml",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="models/Wan2.2-T2V-A14B",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    p.add_argument("--shift", type=float, default=12.0)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--transformer_path", type=str, default=None)
    p.add_argument("--transformer_high_path", type=str, default=None)
    p.add_argument("--vae_path", type=str, default=None)

    # memory / speed
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
    p.add_argument("--compile_dit", action="store_true")
    p.add_argument("--fsdp_dit", action="store_true")
    p.add_argument("--fsdp_text_encoder", action="store_true", default=True)
    p.add_argument("--ulysses_degree", type=int, default=1)
    p.add_argument("--ring_degree", type=int, default=1)

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

    # dtypes
    p.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--text_encoder_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--vae_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    # TeaCache / cfg skip / riflex
    p.add_argument("--enable_teacache", action="store_true", default=True)
    p.add_argument("--teacache_threshold", type=float, default=0.10)
    p.add_argument("--teacache_skip_start", type=int, default=5)
    p.add_argument("--teacache_offload", action="store_true", default=False)
    p.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    p.add_argument("--enable_riflex", action="store_true", default=False)
    p.add_argument("--riflex_k", type=int, default=6)

    # LoRA
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--lora_weight", type=float, default=0.55)
    p.add_argument("--lora_high_path", type=str, default=None)
    p.add_argument("--lora_high_weight", type=float, default=0.55)

    # misc
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


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


_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = _slug_re.sub("_", s)
    return s[:200]


def pick_caption(it, caption_field: str) -> tuple[str, str]:
    if caption_field == "base":
        cap = (it.get("caption") or "").strip()
        return cap, "BASE"
    if caption_field == "aug":
        cap = (it.get("caption_aug") or "").strip()
        return cap, "AUG"
    if caption_field == "contra":
        cap = (it.get("caption_contra") or "").strip()
        return cap, "CONTRA"
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
    img = Image.open(path).convert("RGB")
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    img = img.resize((width, height), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def resolve_ref_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    raw_path = os.path.expanduser(raw_path.strip())
    return raw_path if os.path.isabs(raw_path) else os.path.abspath(raw_path)


def resolve_lora_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    path = os.path.expanduser(path.strip())
    candidate = Path(path)
    if candidate.is_dir():
        tensors = sorted(candidate.glob("*.safetensors"))
        if not tensors:
            raise FileNotFoundError(f"LoRA directory '{path}' does not contain *.safetensors files.")
        return str(tensors[0])
    if not candidate.exists() and candidate.with_suffix(".safetensors").exists():
        return str(candidate.with_suffix(".safetensors"))
    return str(candidate)


def build_scheduler(args, config):
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler]
    scheduler_kwargs = dict(OmegaConf.to_container(config["scheduler_kwargs"]))
    if args.sampler in ("Flow_Unipc", "Flow_DPM++"):
        scheduler_kwargs["shift"] = 1
    return scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_kwargs))


def load_models(args, device):
    config = OmegaConf.load(args.config_path)
    weight_dtype = getattr(torch, args.weight_dtype)
    text_encoder_dtype = getattr(torch, args.text_encoder_dtype)
    vae_dtype = getattr(torch, args.vae_dtype)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_dir,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_dir,
            config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    def _maybe_load_checkpoint(model, ckpt_path):
        if not ckpt_path:
            return
        path = os.path.expanduser(ckpt_path)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[ckpt] Loaded {path}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    _maybe_load_checkpoint(transformer, args.transformer_path)
    _maybe_load_checkpoint(transformer_2, args.transformer_high_path)

    chosen_vae_cls = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_vae_cls.from_pretrained(
        os.path.join(args.model_dir, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(device=device, dtype=vae_dtype)
    _maybe_load_checkpoint(vae, args.vae_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_dir, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=text_encoder_dtype,
    )

    scheduler = build_scheduler(args, config)

    pipeline = Wan2_2Pipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        from functools import partial

        transformer.enable_multi_gpus_inference()
        transformer_2.enable_multi_gpus_inference()
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        if args.fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=text_encoder_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)

    if args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        replace_parameters_by_name(transformer_2, ["modulation"], device=device)
        if hasattr(transformer, "freqs"):
            transformer.freqs = transformer.freqs.to(device=device)
        if hasattr(transformer_2, "freqs"):
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
    return pipeline, transformer, transformer_2, vae, config, boundary


def main():
    args = parse_args()
    rank, world_size, local_rank = ddp_init()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    def _log(*m):
        if rank == 0 and args.verbose:
            print(*m, flush=True)

    with open(args.dataset_json, "r") as f:
        data = json.load(f)

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
        )

    if args.max_videos is not None:
        items = items[: args.max_videos]

    sharded = [it for i, it in enumerate(items) if i % world_size == rank]
    _log(f"[Shard] rank {rank} has {len(sharded)} / {len(items)} items")

    ti2v_requested = bool(args.ref_field or args.ref_frame)
    mode_dir = "TI2V" if ti2v_requested else "T2V"
    cap_tag_for_dir = sharded[0]["cap_tag"] if sharded else "BASE"
    if args.prompt_category is not None:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir / args.prompt_category
    else:
        out_root = Path(args.outdir) / mode_dir / cap_tag_for_dir
    out_root.mkdir(parents=True, exist_ok=True)

    pipe, transformer, transformer_2, vae, config, boundary = load_models(args, device=device)

    weight_dtype = getattr(torch, args.weight_dtype)
    vae_dtype = getattr(torch, args.vae_dtype)
    vae_device = resolve_module_device(vae, device)

    coefficients = get_teacache_coefficients(args.model_dir) if args.enable_teacache else None
    if coefficients is not None:
        pipe.transformer.enable_teacache(
            coefficients,
            args.steps,
            args.teacache_threshold,
            num_skip_start_steps=args.teacache_skip_start,
            offload=args.teacache_offload,
        )
        pipe.transformer_2.share_teacache(transformer=pipe.transformer)
    if args.cfg_skip_ratio:
        pipe.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.steps)
        pipe.transformer_2.share_cfg_skip(transformer=pipe.transformer)

    lora_path = resolve_lora_checkpoint(args.lora_path)
    lora_high_path = resolve_lora_checkpoint(args.lora_high_path)
    lora_low_merged = False
    lora_high_merged = False
    if lora_path:
        pipe = merge_lora(
            pipe,
            lora_path,
            args.lora_weight,
            device=device,
            dtype=weight_dtype,
        )
        lora_low_merged = True
    if lora_high_path:
        pipe = merge_lora(
            pipe,
            lora_high_path,
            args.lora_high_weight,
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )
        lora_high_merged = True

    static_ref_video = None
    if args.ref_frame:
        ref_path = resolve_ref_path(args.ref_frame)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        static_ref_video = load_ref_image_as_video(ref_path, args.height, args.width, vae_device, vae_dtype)
        log(rank, f"[TI2V] Loaded reference frame {ref_path} ({args.ti2v_mode})")

    base_gen = torch.Generator(device=device)

    T = args.num_frames
    tcr = vae.config.temporal_compression_ratio
    if T != 1:
        T = int((T - 1) // tcr * tcr + 1)
    latent_frames = (T - 1) // tcr + 1 if T != 1 else 1
    if args.enable_riflex:
        pipe.transformer.enable_riflex(k=args.riflex_k, L_test=latent_frames)
        pipe.transformer_2.enable_riflex(k=args.riflex_k, L_test=latent_frames)

    for it in sharded:
        prompt = it["caption"]
        cap_tag = it["cap_tag"]
        if args.prompt_category is not None:
            out_root = Path(args.outdir) / mode_dir / cap_tag / args.prompt_category
        else:
            out_root = Path(args.outdir) / mode_dir / cap_tag
        out_root.mkdir(parents=True, exist_ok=True)

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
            idx = 1
            while True:
                candidate = out_root / f"{name_with_rank}_{idx}.mp4"
                if not candidate.exists():
                    out_path = candidate
                    break
                idx += 1

        gen = base_gen.manual_seed(args.seed + it["id"])
        with torch.no_grad():
            item_ref_video = None
            if static_ref_video is not None:
                item_ref_video = static_ref_video
            elif args.ref_field:
                resolved = resolve_ref_path(it.get("ref_value", ""))
                if not resolved or not os.path.exists(resolved):
                    _log(f"[Skip] missing reference for item {it['id']}: {resolved or '(empty)'}")
                    continue
                item_ref_video = load_ref_image_as_video(resolved, args.height, args.width, vae_device, vae_dtype)

            pipe_kwargs = dict(
                prompt=prompt,
                num_frames=T,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                generator=gen,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                boundary=boundary,
                shift=args.shift,
                max_sequence_length=args.max_sequence_length,
            )
            if item_ref_video is not None:
                pipe_kwargs["video"] = item_ref_video
                pipe_kwargs["ti2v_mode"] = args.ti2v_mode

            sample = pipe(**pipe_kwargs).videos

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(sample, str(out_path), fps=args.fps)
        _log(f"[OK] {it['id']:05d} -> {out_path}")

    if lora_low_merged and lora_path:
        pipe = unmerge_lora(
            pipe,
            lora_path,
            args.lora_weight,
            device=device,
            dtype=weight_dtype,
        )
    if lora_high_merged and lora_high_path:
        pipe = unmerge_lora(
            pipe,
            lora_high_path,
            args.lora_high_weight,
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("[Done] All ranks finished.", flush=True)
        dist.destroy_process_group()
    else:
        print("[Done] Single process finished.", flush=True)


if __name__ == "__main__":
    main()
