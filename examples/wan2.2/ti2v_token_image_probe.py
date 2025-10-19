#!/usr/bin/env python3
# ti2v_token_image_probe.py
import argparse
import json
import math
import os
import sys
from types import MethodType
from typing import Dict, List

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from videox_fun.dist import set_multi_gpus_devices
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

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        "Per-token influence and image/text interaction probe (Wan2.2 TI2V)"
    )
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
    p.add_argument(
        "--save-path", type=str, default="samples/wan2.2-ti2v-token-image-probe"
    )
    # token probe
    p.add_argument(
        "--concepts",
        type=str,
        default="",
        help="comma list of words to track (tokenized spans auto-found)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="8,16,24,30",
        help="Comma-separated layer indices to probe",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for attention heatmaps (if you later add them)",
    )
    # compute tweaks
    p.add_argument(
        "--downsample-lat",
        type=int,
        default=1,
        help="optional spatial downsample factor on latents for image/text curves",
    )
    return p.parse_args()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_heatmap(M, title, out_png, xlabel="step", ylabel="token", cmap="viridis"):
    plt.figure(figsize=(10, 4))
    plt.imshow(M, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_curves(
    series_dict: Dict[str, np.ndarray], title, out_png, ylabel="marginal attn"
):
    plt.figure(figsize=(10, 4))
    for name, arr in series_dict.items():
        plt.plot(arr, label=name)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series_dict) <= 16:
        plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def to_bcthw_float(video):
    """
    Accepts:
      - torch.Tensor [B, F, H, W, C] or [B, C, F, H, W] in [0,1] or [0,255]
      - np.ndarray   [B, F, H, W, C] or [B, C, F, H, W] in [0,1] or [0,255]
    Returns:
      torch.float32 tensor [B, 3, T, H, W] in [0,1]
    """
    if isinstance(video, np.ndarray):
        v = torch.from_numpy(video)
    elif isinstance(video, torch.Tensor):
        v = video
    else:
        raise TypeError(f"Unsupported type: {type(video)}")

    if v.dim() != 5:
        raise ValueError(f"Expected 5D video tensor, got {v.shape}")

    # Move channels to dim=1 if needed
    if v.shape[-1] in (1, 3):  # [B,F,H,W,C]
        v = v.permute(0, 4, 1, 2, 3).contiguous()
    elif v.shape[1] in (1, 3):  # [B,C,F,H,W]
        pass
    else:
        raise ValueError(f"Ambiguous channel dim in shape {tuple(v.shape)}")

    v = v.float()
    if v.max() > 1.0:
        v = v / 255.0
    # force 3ch
    if v.shape[1] == 1:
        v = v.repeat(1, 3, 1, 1, 1)
    return v.clamp(0, 1)


# --------- tokenizer span helper ----------
def find_concept_spans(
    tokenizer, prompt: str, concepts: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Returns {concept -> LongTensor(indices in tokenized prompt)} (may be empty)
    """
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
        return_attention_mask=False,
    )
    offsets = enc["offset_mapping"][0].tolist()
    spans = {}
    for w in concepts:
        w_low = w.strip().lower()
        inds = []
        for i, (s, e) in enumerate(offsets):
            piece = prompt[s:e].lower()
            if piece and (w_low in piece):
                inds.append(i)
        if not inds:
            spans[w] = torch.tensor([], dtype=torch.long)
            continue
        # merge contiguous splits -> we still keep them as a flat index set
        spans[w] = torch.tensor(inds, dtype=torch.long)
    return spans


# --------- marginal attention (text-only) ----------
def marginal_attn_for_indices(q, k, idx: torch.Tensor, chunk=256):
    """
    q: [B, Lq, H, D], k: [B, Lt, H, D], idx: [m] subset of {0..Lt-1}
    returns: [B, Lq] = mean over heads of (sum attention weight over idx)
    Softmax taken over tokens dimension. Numerically stable.
    """
    B, Lq, H, D = q.shape
    device = q.device
    q = q.float()
    k = k.float()
    idx = idx.to(device)

    m = idx.numel()
    if m == 0:
        return torch.zeros((B, Lq), device=device, dtype=torch.float32)

    # compute logits over *only* the chosen tokens
    # logits_s = (q @ k_idx^T) / sqrt(D)
    # shape: [B, H, Lq, m]
    k_idx = k[:, idx]  # [B, m, H, D]
    logits = torch.einsum("blhd,bmhd->bhlm", q, k_idx) / math.sqrt(D)

    # we need normalization over the full Lt dimension to get exact marginals;
    # approximate by computing normalization over all tokens in chunks
    denom_max = None
    denom_acc = None

    Lt = k.shape[1]
    for s in range(0, Lt, chunk):
        e = min(s + chunk, Lt)
        k_ch = k[:, s:e]  # [B, c, H, D]
        logits_ch = torch.einsum("blhd,bchd->bhlc", q, k_ch) / math.sqrt(
            D
        )  # [B,H,Lq,c]
        # log-sum-exp accumulation
        if denom_max is None:
            denom_max = logits_ch.max(dim=3, keepdim=True).values  # [B,H,Lq,1]
            denom_acc = torch.exp(logits_ch - denom_max).sum(
                dim=3, keepdim=True
            )  # [B,H,Lq,1]
        else:
            m2 = torch.maximum(denom_max, logits_ch.max(dim=3, keepdim=True).values)
            # rescale accumulators
            denom_acc = denom_acc * torch.exp(denom_max - m2) + torch.exp(
                logits_ch - m2
            ).sum(dim=3, keepdim=True)
            denom_max = m2

    # now numerator over the selected indices
    num = torch.exp(logits - denom_max).sum(dim=3)  # [B,H,Lq]
    denom = denom_acc.squeeze(3) + 1e-8  # [B,H,Lq]
    marg = (num / denom).mean(dim=1)  # mean over heads -> [B,Lq]
    return marg  # in [0,1], per query position


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------


def main():
    args = parse_args()
    ensure_dir(args.save_path)

    # device & dtype
    device = set_multi_gpus_devices(1, 1)
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

    # ---------------------------------------------------------
    # Token spans from prompt (concepts)
    # ---------------------------------------------------------
    if args.concepts.strip():
        concepts = [w.strip() for w in args.concepts.split(",") if w.strip()]
    else:
        # default: use *all* tokens (nice to see which ones matter)
        enc = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        # filter out trivially empty tokens
        concepts = [t for t in toks if t.strip()]
    spans = find_concept_spans(tokenizer, args.prompt, concepts)

    # ---------------------------------------------------------
    # Monkey-patch: collect per-step, per-layer, per-token marginals
    # ---------------------------------------------------------
    # Import the original fast attention function used by Wan
    from videox_fun.models.wan_transformer3d import attention as wan_attention

    layers_to_log = sorted(
        {int(x) for x in args.layers.split(",") if x.strip().isdigit()}
    )
    attn_log = {}  # step -> layer -> {concept -> [B,Lq] Tensor(cpu)}
    current_step = {"idx": -1}  # mutable integer

    def cross_attn_probe(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        """
        Replacement for WanCrossAttention.forward that:
          1) computes marginal attention mass for selected token spans
          2) returns the original cross-attn output
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)  # [B,Lq,H,D]
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)  # [B,Lt,H,D]
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # record marginals (text-only)
        if getattr(self, "_record_attn", False):
            layer_idx = getattr(self, "_layer_idx", None)
            if layer_idx in layers_to_log and current_step["idx"] >= 0:
                step_bucket = attn_log.setdefault(current_step["idx"], {})
                lay_bucket = step_bucket.setdefault(layer_idx, {})
                with torch.cuda.amp.autocast(enabled=False):
                    for name, idxs in self._concept_spans.items():
                        if idxs is None or idxs.numel() == 0:
                            continue
                        m = marginal_attn_for_indices(
                            q.float(), k.float(), idxs, chunk=256
                        )  # [B,Lq]
                        lay_bucket[name] = m.detach().cpu()

        # normal forward
        x_attn = wan_attention(
            q.to(dtype), k.to(dtype), v=v.to(dtype), k_lens=context_lens
        )
        x_attn = x_attn.to(dtype).flatten(2)
        x_out = self.o(x_attn)
        return x_out

    # Attach to each WanAttentionBlock.cross_attn (class WanCrossAttention in Wan2.2 TI2V)
    for li, blk in enumerate(pipe.transformer.blocks):
        if (
            hasattr(blk, "cross_attn")
            and blk.cross_attn.__class__.__name__ == "WanCrossAttention"
        ):
            ca = blk.cross_attn
            ca._record_attn = True
            ca._layer_idx = li
            ca._concept_spans = spans
            ca.forward = MethodType(cross_attn_probe, ca)

    # ---------------------------------------------------------
    # Prepare masked-image latents for image/text interaction curves (5D for VAE)
    # ---------------------------------------------------------
    with torch.no_grad():
        # input_video: [B,3,Fpix,Hpix,Wpix] in [0,1]
        # input_mask : [B,1,Fpix,Hpix,Wpix] in {0,1}
        B, C, Fpix, Hpix, Wpix = input_video.shape
        mask_condition = input_mask.to(dtype=torch.float32)  # [B,1,Fpix,H,W]
        masked_video = input_video * (mask_condition < 0.5)  # keep image where mask==0

        # --- Encode masked video in 5D, as expected by Wan VAE ---
        mv5 = masked_video.to(pipe.vae.dtype).to(device)  # [B,3,Fpix,H,W]
        m_lat_5d = pipe.vae.encode(mv5)[0].mode()  # [B, C_lat, F_lat, H/8, W/8]
        C_lat, F_lat, Hlat, Wlat = (
            m_lat_5d.shape[1],
            m_lat_5d.shape[2],
            m_lat_5d.shape[3],
            m_lat_5d.shape[4],
        )

        # --- Down/align the mask to latent resolution/time ---
        # Resize mask to [B,1,F_lat,H/8,W/8] using trilinear (float), then binarize
        msk_lat = F.interpolate(
            mask_condition,
            size=(
                F_lat,
                Hpix // pipe.vae.spatial_compression_ratio,
                Wpix // pipe.vae.spatial_compression_ratio,
            ),
            mode="trilinear",
            align_corners=False,
        ).to(device=mv5.device, dtype=pipe.text_encoder.dtype)
        msk_lat = (msk_lat > 0.5).float()  # [B,1,F_lat,H/8,W/8]

        # If you downsample latents later for speed, also downsample m_lat_5d and msk_lat consistently.
        m_lat = m_lat_5d  # keep name used below

    # ---------------------------------------------------------
    # Callback: per-step bookkeeping
    #   - set current_step
    #   - compute image/text interaction numbers
    # ---------------------------------------------------------
    steps_seen = []
    img_text_curve = {"masked_L2": [], "unmasked_L2": []}  # per step scalars

    lat_down = max(1, int(args.downsample_lat))

    def on_step_end(pipeline, i, t, kwargs):
        current_step["idx"] = int(i)
        steps_seen.append(int(i))

        L = kwargs["latents"]  # [B,C_lat,F_lat,H/8,W/8]
        if lat_down > 1:
            Ls = F.avg_pool3d(
                L, kernel_size=(1, lat_down, lat_down), stride=(1, lat_down, lat_down)
            )
            Ms = F.avg_pool3d(
                msk_lat.to(L.device),
                kernel_size=(1, lat_down, lat_down),
                stride=(1, lat_down, lat_down),
            )
            Mlat = (Ms > 0.5).float()
        else:
            Ls = L
            Mlat = (msk_lat.to(L.device) > 0.5).float()

        mlat_ref = m_lat.to(Ls.device)
        if lat_down > 1:
            mlat_ref = F.avg_pool3d(
                mlat_ref,
                kernel_size=(1, lat_down, lat_down),
                stride=(1, lat_down, lat_down),
            )

        diff = (Ls - mlat_ref).float().pow(2).sqrt()  # [B,C,F,H,W]
        # masked vs unmasked mean L2
        masked_L2 = (diff * Mlat).sum() / (Mlat.sum() + 1e-6)
        unmasked_L2 = (diff * (1 - Mlat)).sum() / ((1 - Mlat).sum() + 1e-6)
        img_text_curve["masked_L2"].append(float(masked_L2.detach().cpu()))
        img_text_curve["unmasked_L2"].append(float(unmasked_L2.detach().cpu()))

        return {"latents": kwargs["latents"]}

    # ---------------------------------------------------------
    # Run generation (request "pt" and save directly)
    # ---------------------------------------------------------
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
            output_type="numpy",
        )

    video_pt = out.videos  # torch [B,C,T,H,W] float in [0,1]
    vid_path = os.path.join(args.save_path, "result.mp4")
    save_videos_grid(video_pt, vid_path, fps=args.fps)

    # ---------------------------------------------------------
    # Reduce attention logs -> per-token curves over steps
    # We average marginal mass over all query positions (Lq) for a compact curve.
    # ---------------------------------------------------------
    # attn_log: step -> layer -> {concept -> [B,Lq]}
    token_series = {}  # concept -> np.array[S] averaged over layers you picked
    steps = sorted(attn_log.keys())
    if steps:
        # collect per-concept per-step means (avg over Lq, avg over layers)
        for name in concepts:
            per_step_vals = []
            for s in steps:
                lay_dict = attn_log.get(s, {})
                vals_this_step = []
                for L in layers_to_log:
                    if L in lay_dict and name in lay_dict[L]:
                        A = lay_dict[L][name]  # [B,Lq] cpu
                        vals_this_step.append(A.mean().item())
                if vals_this_step:
                    per_step_vals.append(float(np.mean(vals_this_step)))
                else:
                    per_step_vals.append(0.0)
            token_series[name] = np.array(per_step_vals, dtype=np.float32)

        # save curves
        save_curves(
            token_series,
            "Token marginal attention (avg over layers)",
            os.path.join(args.save_path, "token_attention_curves.png"),
            ylabel="marginal attn",
        )

        # also save a 2D heatmap (tokens as rows)
        if token_series:
            M = np.stack([token_series[k] for k in concepts], axis=0)  # [tokens, steps]
            save_heatmap(
                M,
                "Token marginal attention heatmap",
                os.path.join(args.save_path, "token_attention_heatmap.png"),
                xlabel="step",
                ylabel="token",
            )

    # ---------------------------------------------------------
    # Save image/text interaction curves
    # ---------------------------------------------------------
    save_curves(
        {
            "masked_L2": np.array(img_text_curve["masked_L2"]),
            "unmasked_L2": np.array(img_text_curve["unmasked_L2"]),
        },
        "Image vs Text influence (latent L2 to masked-image latents)",
        os.path.join(args.save_path, "image_text_interaction_curves.png"),
        ylabel="mean L2",
    )

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------
    manifest = {
        "prompt": args.prompt,
        "image": args.image,
        "height": args.height,
        "width": args.width,
        "video_length": args.video_length,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "shift": args.shift,
        "layers": layers_to_log,
        "concepts": concepts,
        "result_video": vid_path,
        "steps_seen": steps_seen,
    }
    with open(os.path.join(args.save_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Outputs in: {args.save_path}")


if __name__ == "__main__":
    main()


# python examples/wan2.2/ti2v_token_image_probe.py \
#   --prompt "a small orange elephant running in the field" \
#   --image /capstor/scratch/cscs/mhasan/VideoX-Fun-ours/test_assets/premium_photo-1669277330871-443462026e13.jpeg \
#   --concepts small,orange,elephant,running,field \
#   --layers 8,16,24,30 \
#   --height 704 --width 1280 --video-length 121 \
#   --num-steps 50 --guidance-scale 6.0 --sampler Flow_Unipc --shift 5 \
#   --save-path samples/wan2.2-ti2v-token-image-probe
