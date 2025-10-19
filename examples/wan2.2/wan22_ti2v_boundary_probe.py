#!/usr/bin/env python3
# detect_step_change_ti2v.py
import argparse
import json
import math
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# --- project roots (adjust if needed) ---
this = os.path.abspath(__file__)
for r in [
    os.path.dirname(this),
    os.path.dirname(os.path.dirname(this)),
    os.path.dirname(os.path.dirname(os.path.dirname(this))),
]:
    if r not in sys.path:
        sys.path.insert(0, r)

# --- videox_fun / diffusers imports (Wan2.2 TI2V) ---
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
from videox_fun.pipeline import Wan2_2TI2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)


def parse_args():
    p = argparse.ArgumentParser("Step-wise change detector for TI2V (latents only)")
    # core gen
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--image", type=str, required=True, help="reference image path")
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--video-length", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument(
        "--negative-prompt",
        type=str,
        default="overexposed, blurry, low quality, artifacts, static scene",
    )
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument(
        "--sampler",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    p.add_argument("--shift", type=float, default=5.0, help="used by Flow_Unipc/DPM++")
    # model
    p.add_argument("--config", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    p.add_argument("--model", type=str, default="models/Wan2.2-TI2V-5B")
    # logging / out
    p.add_argument("--save-path", type=str, default="samples/wan2.2-ti2v-step-change")
    # detector options
    p.add_argument(
        "--reference",
        type=str,
        default="prev",
        choices=["prev", "first"],
        help="compare each step to previous step or to the very first step",
    )
    p.add_argument(
        "--metric", type=str, default="both", choices=["cosine", "l2", "both"]
    )
    p.add_argument(
        "--frame-indices",
        type=str,
        default="all",
        help="comma list of frame indices at decoded rate (e.g., 0,10,20) or 'all'",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="optional spatial downsample factor on latents",
    )
    return p.parse_args()


# ------------- metrics on latents -------------
def spatial_pool(x):
    # x: [B,C,F,H,W] -> [B,C,F]
    return x.mean(dim=(-1, -2))


def per_frame_cosine(prev, cur, eps=1e-8):
    # prev,cur: [B,C,F]
    prev = prev.float()
    cur = cur.float()
    prev_n = prev / (prev.norm(dim=1, keepdim=True) + eps)
    cur_n = cur / (cur.norm(dim=1, keepdim=True) + eps)
    cos = (prev_n * cur_n).sum(dim=1)  # [B,F]
    return 1.0 - cos  # cosine distance per frame


def per_frame_l2(prev, cur):
    # prev,cur: [B,C,F]
    return (cur - prev).pow(2).mean(dim=1).sqrt()  # [B,F]


# ------------- plotting / saving -------------
def save_heatmap(M, title, out_png, xlabel="frame", ylabel="step", cmap="viridis"):
    plt.figure(figsize=(10, 4))
    plt.imshow(M, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_curves(per_step_per_frame, frame_ids, title, out_png, ylabel):
    plt.figure(figsize=(10, 4))
    for fi in frame_ids:
        plt.plot(per_step_per_frame[:, fi], label=f"f{fi}")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(frame_ids) <= 12:
        plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def to_bcthw_uint8(video):
    """
    Accepts:
      - torch.Tensor [B, F, H, W, C] or [B, C, F, H, W] in [0,1] or [0,255]
      - np.ndarray   [B, F, H, W, C] or [B, C, F, H, W] in [0,1] or [0,255]
    Returns:
      torch.uint8 tensor [B, 3, T, H, W] (channels-first)
    """
    import numpy as np
    import torch

    # to torch
    if isinstance(video, np.ndarray):
        v = torch.from_numpy(video)
    elif isinstance(video, torch.Tensor):
        v = video
    else:
        raise TypeError(f"Unsupported type: {type(video)}")

    # ensure float for scaling
    if v.dtype == torch.uint8:
        v8 = v
    else:
        v = v.float()
        # guess scale
        if v.max() <= 1.0:
            v = v * 255.0
        v8 = v.clamp(0, 255).to(torch.uint8)

    # unify shapes to [B, C, T, H, W] with C last check
    if v8.dim() != 5:
        raise ValueError(f"Expected 5D video tensor, got {v8.shape}")

    B, A, B2, C2, D2 = v8.shape  # just for thinking, not used directly

    # cases:
    # [B, F, H, W, C]  -> permute(0, 4, 1, 2, 3)
    # [B, C, F, H, W]  -> already fine
    if v8.shape[-1] in (1, 3):  # ...C at the end
        v8 = v8.permute(0, 4, 1, 2, 3).contiguous()
    elif v8.shape[1] in (1, 3):  # C already in dim=1
        pass
    else:
        raise ValueError(
            f"Cannot infer channel dim. Got shape {tuple(v8.shape)}; "
            "need either [B,F,H,W,C] or [B,C,F,H,W]."
        )

    # swap single-channel to 3-ch if needed
    if v8.shape[1] == 1:
        v8 = v8.repeat(1, 3, 1, 1, 1)

    # sanity checks
    B, C, T, H, W = v8.shape
    if C != 3:
        raise ValueError(f"Channels must be 3, got {C}")
    return v8


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # device & dtype
    GPU_memory_mode = "sequential_cpu_offload"
    ulysses_degree = 1
    ring_degree = 1
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    weight_dtype = torch.bfloat16

    # --- load cfg + parts
    config = OmegaConf.load(args.config)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer_2 = None
    if (
        config["transformer_additional_kwargs"].get(
            "transformer_combination_type", "single"
        )
        == "moe"
    ):
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                args.model,
                config["transformer_additional_kwargs"].get(
                    "transformer_high_noise_model_subpath", "transformer"
                ),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

    Chosen_VAE = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_VAE.from_pretrained(
        os.path.join(args.model, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # scheduler
    if args.sampler == "Flow":
        Scheduler = FlowMatchEulerDiscreteScheduler
    elif args.sampler == "Flow_Unipc":
        Scheduler = FlowUniPCMultistepScheduler
        config["scheduler_kwargs"]["shift"] = 1
    else:
        Scheduler = FlowDPMSolverMultistepScheduler
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = Scheduler(
        **filter_kwargs(Scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    # pipeline
    pipe = Wan2_2TI2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # memory mode
    replace_parameters_by_name(transformer, ["modulation"], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation"], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipe.enable_sequential_cpu_offload(device=device)

    # TeaCache / cfg-skip (optional)
    coeffs = get_teacache_coefficients(args.model)
    if coeffs is not None:
        pipe.transformer.enable_teacache(
            coeffs, args.num_steps, 0.10, num_skip_start_steps=5, offload=False
        )
        if transformer_2 is not None:
            pipe.transformer_2.share_teacache(transformer=pipe.transformer)
    pipe.transformer.enable_cfg_skip(0, args.num_steps)
    if transformer_2 is not None:
        pipe.transformer_2.share_cfg_skip(transformer=pipe.transformer)

    # seed
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # input video latent from image
    video_length = (
        int(
            (args.video_length - 1)
            // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio
        )
        + 1
        if args.video_length != 1
        else 1
    )
    input_video, input_mask, _ = get_image_to_video_latent(
        args.image,
        None,
        video_length=video_length,
        sample_size=[args.height, args.width],
    )

    # frame selection
    if args.frame_indices.strip().lower() == "all":
        frame_indices = None  # will use all frames
    else:
        frame_indices = [int(i) for i in args.frame_indices.split(",") if i.strip()]

    # containers for step-wise metrics
    lat_down = args.downsample
    cos_records = []
    l2_records = []
    steps_seen = []

    ref_lat = {"first": None, "prev": None}

    def on_step_end(pipeline, i, t, kwargs):
        # kwargs has: 'latents', 'prompt_embeds', 'negative_prompt_embeds'
        L = kwargs["latents"]  # [B,C,F',H',W'] (latent frame-rate)
        # optional spatial downsample for speed
        if lat_down > 1:
            Ld = torch.nn.functional.avg_pool3d(
                L, kernel_size=(1, lat_down, lat_down), stride=(1, lat_down, lat_down)
            )
        else:
            Ld = L

        # pool to [B,C,F]
        Lp = spatial_pool(Ld)  # [B,C,F_lat]
        if frame_indices is not None:
            keep = torch.tensor(frame_indices, device=Lp.device)
            keep = keep.clamp_max(Lp.shape[-1] - 1)
            Lp = Lp.index_select(-1, keep)

        # choose reference
        if args.reference == "first":
            if ref_lat["first"] is None:
                ref_lat["first"] = Lp.detach().clone()
                # no metric at step 0 against itself; store zeros
                fsz = Lp.shape[-1]
                if args.metric in ("cosine", "both"):
                    cos_records.append(torch.zeros((Lp.shape[0], fsz)))
                if args.metric in ("l2", "both"):
                    l2_records.append(torch.zeros((Lp.shape[0], fsz)))
            else:
                if args.metric in ("cosine", "both"):
                    cos_d = per_frame_cosine(ref_lat["first"], Lp).detach().cpu()
                    cos_records.append(cos_d)
                if args.metric in ("l2", "both"):
                    l2_d = per_frame_l2(ref_lat["first"], Lp).detach().cpu()
                    l2_records.append(l2_d)
        else:  # prev
            if ref_lat["prev"] is None:
                ref_lat["prev"] = Lp.detach().clone()
                fsz = Lp.shape[-1]
                if args.metric in ("cosine", "both"):
                    cos_records.append(torch.zeros((Lp.shape[0], fsz)))
                if args.metric in ("l2", "both"):
                    l2_records.append(torch.zeros((Lp.shape[0], fsz)))
            else:
                if args.metric in ("cosine", "both"):
                    cos_d = per_frame_cosine(ref_lat["prev"], Lp).detach().cpu()
                    cos_records.append(cos_d)
                if args.metric in ("l2", "both"):
                    l2_d = per_frame_l2(ref_lat["prev"], Lp).detach().cpu()
                    l2_records.append(l2_d)
            ref_lat["prev"] = Lp.detach().clone()

        steps_seen.append(int(i))
        return {"latents": kwargs["latents"]}  # must return latents at least

    # run generation with callback
    with torch.no_grad():
        out = pipe(
            args.prompt,
            num_frames=video_length,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            generator=gen,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_mask,
            shift=args.shift,
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=[
                "latents",
                "prompt_embeds",
                "negative_prompt_embeds",
            ],
            output_type="numpy",  # we still decode once at the end
        )
    video = out.videos  # [B,F,H,W,3] in [0,1] np

    # # save result video
    vid_path = os.path.join(args.save_path, "result.mp4")

    # vid_tensor = to_bcthw_uint8(video)

    # print("vid_tensor", vid_tensor.shape, vid_tensor.dtype)

    # Now this matches your utils signature
    save_videos_grid(video, vid_path, fps=args.fps)

    # stack metrics -> [S,B,F_sel] -> we’ll take mean over batch (usually B=1)
    S = len(steps_seen)
    if args.metric in ("cosine", "both"):
        C = torch.stack(cos_records, dim=0)  # [S,B,F]
        C_mean = C.mean(dim=1).numpy()  # [S,F]
        np.save(os.path.join(args.save_path, "cosine_per_step_frame.npy"), C_mean)
        # heatmap and curves
        save_heatmap(
            C_mean,
            f"Cosine distance vs {args.reference} (latents)",
            os.path.join(args.save_path, "heatmap_cosine.png"),
        )
        # curves for a few frames (or all if few)
        if C_mean.shape[1] <= 16:
            frame_ids = list(range(C_mean.shape[1]))
        else:
            # pick uniform 10 frames
            frame_ids = np.linspace(0, C_mean.shape[1] - 1, num=10, dtype=int).tolist()
        save_curves(
            C_mean,
            frame_ids,
            f"Cosine distance vs {args.reference}",
            os.path.join(args.save_path, "curves_cosine.png"),
            ylabel="1 - cos",
        )

    if args.metric in ("l2", "both"):
        L2 = torch.stack(l2_records, dim=0)  # [S,B,F]
        L2_mean = L2.mean(dim=1).numpy()
        np.save(os.path.join(args.save_path, "l2_per_step_frame.npy"), L2_mean)
        save_heatmap(
            L2_mean,
            f"L2 vs {args.reference} (latents)",
            os.path.join(args.save_path, "heatmap_l2.png"),
        )
        if L2_mean.shape[1] <= 16:
            frame_ids = list(range(L2_mean.shape[1]))
        else:
            frame_ids = np.linspace(0, L2_mean.shape[1] - 1, num=10, dtype=int).tolist()
        save_curves(
            L2_mean,
            frame_ids,
            f"L2 vs {args.reference}",
            os.path.join(args.save_path, "curves_l2.png"),
            ylabel="L2",
        )

    # save a small manifest
    manifest = {
        "prompt": args.prompt,
        "image": args.image,
        "height": args.height,
        "width": args.width,
        "video_length": args.video_length,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "reference": args.reference,
        "metric": args.metric,
        "frame_indices": args.frame_indices,
        "downsample": args.downsample,
        "steps_seen": steps_seen,
        "result_video": vid_path,
    }
    with open(os.path.join(args.save_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Outputs in: {args.save_path}")


if __name__ == "__main__":
    main()


# python examples/wan2.2/wan22_ti2v_boundary_probe.py \
#   --prompt "a cat running in the field" \
#   --image /capstor/scratch/cscs/mhasan/VideoX-Fun-ours/test_assets/premium_photo-1669277330871-443462026e13.jpeg \
#   --save-path samples/wan2.2-ti2v-event-change
