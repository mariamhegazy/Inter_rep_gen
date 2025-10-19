#!/usr/bin/env python3
# ti2v_token_frame_attn.py
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

# Wan2.2 TI2V imports
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


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Visualize per-token attention across frames (Wan2.2 TI2V)"
    )
    # generation
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
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
    p.add_argument("--shift", type=float, default=5.0)

    # model
    p.add_argument("--config", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    p.add_argument("--model", type=str, default="models/Wan2.2-TI2V-5B")

    # logging / out
    p.add_argument("--save-path", type=str, default="samples/wan2.2-token-frame-attn")

    # attention capture
    p.add_argument(
        "--layers",
        type=str,
        default="24,28,30",
        help="comma list of cross-attn layer indices to log",
    )
    p.add_argument(
        "--concepts",
        type=str,
        default="",
        help="comma words to track; if empty, use all tokens",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    p.add_argument(
        "--steps",
        type=str,
        default="last",
        help="'last' | 'all' | comma indices e.g. '0,10,20,49'",
    )
    p.add_argument(
        "--avg-layers",
        action="store_true",
        help="average selected layers before visualization",
    )

    # compute tweaks
    p.add_argument(
        "--downsample-lat",
        type=int,
        default=1,
        help="optional latent spatial downsample for metrics",
    )
    return p.parse_args()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------
# Token spans
# -----------------------
def find_concept_spans(
    tokenizer, prompt: str, concepts: List[str]
) -> Dict[str, torch.Tensor]:
    import torch

    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)

    def subseq(a, b):
        L, M = len(a), len(b)
        for i in range(L - M + 1):
            if a[i : i + M] == b:
                return list(range(i, i + M))
        return []

    def norm(piece):  # strip SentencePiece underline and lowercase
        return piece.replace("▁", "").lower()

    spans = {}
    for w in concepts:
        w = w.strip()
        if not w:
            spans[w] = torch.tensor([], dtype=torch.long)
            continue

        w_ids = tokenizer(w, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].tolist()
        hit = subseq(ids, w_ids) if w_ids else []
        if not hit:
            # fuzzy fallback
            target = w.lower()
            hits = [i for i, p in enumerate(toks) if target in norm(p)]
            if hits:
                # choose the longest contiguous run
                runs, cur = [], [hits[0]]
                for a, b in zip(hits, hits[1:]):
                    if b == a + 1:
                        cur.append(b)
                    else:
                        runs.append(cur)
                        cur = [b]
                runs.append(cur)
                hit = max(runs, key=len)
        spans[w] = torch.tensor(hit, dtype=torch.long)
    return spans


# -----------------------
# Marginal attention
# -----------------------
def marginal_attn_for_indices(q, k, idx: torch.Tensor, chunk=256):
    """
    q: [B,Lq,H,D], k: [B,Lt,H,D], idx: [m in 0..Lt-1]
    -> [B,Lq] mean over heads of attention mass on idx
    """
    B, Lq, H, D = q.shape
    device = q.device
    q = q.float()
    k = k.float()
    idx = idx.to(device)
    if idx.numel() == 0:
        return torch.zeros((B, Lq), device=device, dtype=torch.float32)

    k_idx = k[:, idx]  # [B,m,H,D]
    logits_sel = torch.einsum("blhd,bmhd->bhlm", q, k_idx) / math.sqrt(D)  # [B,H,Lq,m]

    denom_max, denom_acc = None, None
    Lt = k.shape[1]
    for s in range(0, Lt, chunk):
        e = min(s + chunk, Lt)
        k_ch = k[:, s:e]
        logits_ch = torch.einsum("blhd,bchd->bhlc", q, k_ch) / math.sqrt(D)
        if denom_max is None:
            denom_max = logits_ch.max(dim=3, keepdim=True).values
            denom_acc = torch.exp(logits_ch - denom_max).sum(dim=3, keepdim=True)
        else:
            m2 = torch.maximum(denom_max, logits_ch.max(dim=3, keepdim=True).values)
            denom_acc = denom_acc * torch.exp(denom_max - m2) + torch.exp(
                logits_ch - m2
            ).sum(dim=3, keepdim=True)
            denom_max = m2

    num = torch.exp(logits_sel - denom_max).sum(dim=3)  # [B,H,Lq]
    den = denom_acc.squeeze(3) + 1e-8  # [B,H,Lq]
    marg = (num / den).mean(dim=1)  # [B,Lq]
    return marg


# -----------------------
# Upsample Lq -> frames
# -----------------------
def marginals_to_frames(
    A_flat_B_Lq, Fp, Hp, Wp, F_tgt, H, W, norm="frame", clip=(0.01, 0.99)
):
    # A_flat_B_Lq: [B, Lq] (CPU or CUDA) -> returns [B, F_tgt, H, W] on CPU
    A = A_flat_B_Lq.to(torch.float32).cpu().contiguous()
    B, Lq = A.shape
    assert Lq == Fp * Hp * Wp, f"Lq={Lq} != Fp*Hp*Wp={Fp*Hp*Wp}"

    # reshape to latent grid and upsample to frame space
    A = A.view(B, Fp, Hp, Wp).contiguous()
    A_up = F.interpolate(
        A.unsqueeze(1), size=(F_tgt, H, W), mode="trilinear", align_corners=False
    ).squeeze(
        1
    )  # [B, F_tgt, H, W]

    # ---- robust percentile clipping (per-frame to avoid huge tensors)
    lo, hi = clip
    if (lo, hi) != (0.0, 1.0):
        # shape -> [B, F_tgt, H*W] then quantile over last dim
        flat = A_up.flatten(2)  # [B, F_tgt, H*W]
        # quantiles per frame
        q_lo = torch.quantile(flat, lo, dim=2, keepdim=True)
        q_hi = torch.quantile(flat, hi, dim=2, keepdim=True)
        # normalize using per-frame percentiles
        A_up = (flat - q_lo).clamp(min=0) / (q_hi - q_lo + 1e-6)
        A_up = A_up.view(B, F_tgt, H, W)

    # additional normalization
    if norm == "frame":
        mins = A_up.amin(dim=(2, 3), keepdim=True)
        maxs = A_up.amax(dim=(2, 3), keepdim=True)
        A_up = (A_up - mins) / (maxs - mins + 1e-6)
    elif norm == "video":
        A_up = A_up - A_up.amin(dim=(1, 2, 3), keepdim=True)
        A_up = A_up / (A_up.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    # elif norm == "none": pass
    # elif norm == "log": you can add log1p if you want

    return A_up


def overlay_heatmap(video_uint8_FHWC, heat_FHW, alpha=0.45):
    """
    video_uint8_FHWC: uint8 [F,H,W,3]
    heat_FHW: float [F,H,W] in [0,1]
    returns uint8 [F,H,W,3]
    """
    import cv2

    F, H, W, _ = video_uint8_FHWC.shape
    out = []
    for f in range(F):
        base = video_uint8_FHWC[f]
        h = (heat_FHW[f] * 255).astype(np.uint8)
        h = cv2.applyColorMap(h, cv2.COLORMAP_JET)[:, :, ::-1]  # to RGB
        over = (alpha * h + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
        out.append(over)
    return np.stack(out, axis=0)


def to_bcthw_uint8(vid_FHWC):
    v = (
        torch.from_numpy(vid_FHWC).permute(1, 4, 0, 2, 3)
        if vid_FHWC.ndim == 5
        else torch.from_numpy(vid_FHWC).permute(3, 0, 1, 2).unsqueeze(0)
    )
    if v.max() <= 1:
        v = v * 255
    return v.clamp(0, 255).to(torch.uint8)


def as_numpy_video_FHWC(x):
    """
    Return np.ndarray of shape [F, H, W, 3] in [0, 255] uint8.
    Accepts:
      - np.ndarray [F,H,W,3] in [0,1] or [0,255]
      - torch.Tensor [B,F,H,W,3] or [B,3,F,H,W] or [F,H,W,3] in [0,1]
      - np.ndarray [B,F,H,W,3] or [B,3,F,H,W]
    """
    import numpy as np
    import torch

    # to numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim == 5:
        # assume batch 1
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {x.shape}")
        x = x[0]

    # Now 4D. Normalize to [F,H,W,3].
    if x.ndim != 4:
        raise ValueError(f"Expected 4D video, got shape {x.shape}")

    if x.shape[-1] == 3:
        # [F,H,W,3] or [H,W,F,3]
        # Decide by checking which axis equals the number of frames later.
        # Heuristic: frames should be the axis whose length matches your decoded length
        # but if unknown, assume it's already [F,H,W,3].
        v = x
        # sanity: ensure last dim really is channels
        if v.dtype != np.uint8:
            v = (v * (255.0 if v.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
        return v

    if x.shape[0] == 3:
        # [3,F,H,W] or [3,H,W,F] -> move channels-last
        # Most likely [3,F,H,W]
        v = x.transpose(1, 2, 3, 0)  # -> [F,H,W,3]
        if v.dtype != np.uint8:
            v = (v * (255.0 if v.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
        return v

    # If we get here, try to infer: whichever axis has length 3 is channels.
    ch_axis = [i for i, s in enumerate(x.shape) if s == 3]
    if not ch_axis:
        raise ValueError(f"Cannot find channel axis of size 3 in shape {x.shape}")
    ch = ch_axis[0]
    # move channel axis to the end
    perm = [i for i in range(4) if i != ch] + [ch]
    v = x.transpose(*perm)
    if v.dtype != np.uint8:
        v = (v * (255.0 if v.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
    return v


def ensure_heat_FHW(heat, F, H, W):
    """
    Make sure heat is float32 [F,H,W] in [0,1].
    """
    import numpy as np
    import torch

    if isinstance(heat, torch.Tensor):
        heat = heat.detach().cpu().numpy()
    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim != 3:
        raise ValueError(f"Expected heat shape [F,H,W], got {heat.shape}")
    if heat.shape != (F, H, W):
        # final safeguard: resize per-frame
        import cv2

        out = np.zeros((F, H, W), dtype=np.float32)
        for f in range(min(F, heat.shape[0])):
            out[f] = cv2.resize(heat[f], (W, H), interpolation=cv2.INTER_LINEAR)
        heat = out
    heat = np.clip(heat, 0.0, 1.0)
    return heat


def save_np_FHWC_video(np_fhwc, path, fps):
    """
    np_fhwc: [F,H,W,3] uint8 OR float [0,1]
    """
    import numpy as np
    import torch

    v = np_fhwc
    if v.dtype == np.uint8:
        v = v.astype(np.float32) / 255.0
    else:
        v = v.astype(np.float32)
        if v.max() > 1.0:
            v = np.clip(v / 255.0, 0.0, 1.0)

    # to [1,3,F,H,W] float in [0,1]
    vt = torch.from_numpy(v.transpose(3, 0, 1, 2)[None])
    save_videos_grid(vt, path, fps=fps)


# -----------------------
# main
# -----------------------
def main():
    args = parse_args()
    ensure_dir(args.save_path)

    # device
    device = set_multi_gpus_devices(1, 1)
    weight_dtype = torch.bfloat16

    # load config + parts
    cfg = OmegaConf.load(args.config)
    boundary = cfg["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model,
            cfg["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            cfg["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer_2 = None
    if (
        cfg["transformer_additional_kwargs"].get(
            "transformer_combination_type", "single"
        )
        == "moe"
    ):
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                args.model,
                cfg["transformer_additional_kwargs"].get(
                    "transformer_high_noise_model_subpath", "transformer"
                ),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                cfg["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

    ChosenVAE = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[cfg["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = ChosenVAE.from_pretrained(
        os.path.join(args.model, cfg["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model, cfg["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model,
            cfg["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # scheduler
    if args.sampler == "Flow":
        Scheduler = FlowMatchEulerDiscreteScheduler
    elif args.sampler == "Flow_Unipc":
        Scheduler = FlowUniPCMultistepScheduler
        cfg["scheduler_kwargs"]["shift"] = 1
    else:
        Scheduler = FlowDPMSolverMultistepScheduler
        cfg["scheduler_kwargs"]["shift"] = 1
    scheduler = Scheduler(
        **filter_kwargs(Scheduler, OmegaConf.to_container(cfg["scheduler_kwargs"]))
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

    # TeaCache (optional ok)
    coeffs = get_teacache_coefficients(args.model)
    if coeffs is not None:
        pipe.transformer.enable_teacache(
            coeffs, args.num_steps, 0.10, num_skip_start_steps=5, offload=False
        )
        if transformer_2 is not None:
            pipe.transformer_2.share_teacache(transformer=pipe.transformer)

    # CFG skip disabled for clarity
    pipe.transformer.enable_cfg_skip(0, args.num_steps)
    if transformer_2 is not None:
        pipe.transformer_2.share_cfg_skip(transformer=pipe.transformer)

    # seed
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # input image -> latent video conditioning
    video_length = (
        (
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

    # concepts/tokens
    if args.concepts.strip():
        concept_list = [w.strip() for w in args.concepts.split(",") if w.strip()]
    else:
        enc = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        concept_list = [t for t in toks if t.strip()]
    spans = find_concept_spans(tokenizer, args.prompt, concept_list)

    non_empty = {k: v.tolist() for k, v in spans.items() if v.numel() > 0}
    empty = [k for k, v in spans.items() if v.numel() == 0]
    print(f"[info] spans found: {non_empty}")
    if empty:
        print(f"[warn] empty spans for: {empty}")
        # optional: write to disk for later inspection
        with open(os.path.join(args.save_path, "spans.json"), "w") as f:
            json.dump({"found": non_empty, "empty": empty}, f, indent=2)

    # layers & steps
    layers_to_log = sorted(
        {int(x) for x in args.layers.split(",") if x.strip().isdigit()}
    )

    step_mode = args.steps.strip().lower()
    step_filter = None
    if step_mode not in ("last", "all"):
        step_filter = sorted(
            {int(x) for x in step_mode.split(",") if x.strip().isdigit()}
        )

    # -------------------------
    # Monkey-patch cross-attn
    # -------------------------
    from videox_fun.models.wan_transformer3d import attention as wan_attention

    attn_log = {}  # step -> layer -> {token -> [B,Lq]}
    current_step = {"idx": -1}

    def cross_attn_probe(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)  # [B,Lq,H,D]
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # --- diffusion step: prefer pipeline’s attribute, else callback fallback
        step_idx = -1
        try:
            step_idx = int(self._get_step())
        except Exception:
            step_idx = current_step["idx"]

        if getattr(self, "_record_attn", False):
            layer_idx = getattr(self, "_layer_idx", None)
            if layer_idx in layers_to_log and step_idx >= 0:
                sb = attn_log.setdefault(step_idx, {})
                lb = sb.setdefault(layer_idx, {})
                with torch.cuda.amp.autocast(enabled=False):
                    for name, idxs in self._concept_spans.items():
                        if idxs is None or idxs.numel() == 0:
                            continue
                        m = marginal_attn_for_indices(
                            q.float(), k.float(), idxs, chunk=256
                        )  # [B,Lq]
                        lb[name] = m.detach().cpu()

        from videox_fun.models.wan_transformer3d import attention as wan_attention

        x_attn = wan_attention(
            q.to(dtype), k.to(dtype), v=v.to(dtype), k_lens=context_lens
        )
        x_attn = x_attn.to(dtype).flatten(2)
        return self.o(x_attn)

        from videox_fun.models.wan_transformer3d import attention as wan_attention

        x_attn = wan_attention(
            q.to(dtype), k.to(dtype), v=v.to(dtype), k_lens=context_lens
        )
        x_attn = x_attn.to(dtype).flatten(2)
        return self.o(x_attn)

    # --- inject step getter + layer info + spans; bind forward ---
    for li, blk in enumerate(pipe.transformer.blocks):
        if (
            hasattr(blk, "cross_attn")
            and blk.cross_attn.__class__.__name__ == "WanCrossAttention"
        ):
            ca = blk.cross_attn
            ca._record_attn = True
            ca._layer_idx = li
            ca._concept_spans = spans
            # capture a reference to the *same* transformer the pipeline updates each step
            ca._get_step = lambda tr=pipe.transformer: getattr(tr, "current_steps", -1)
            from types import MethodType

            ca.forward = MethodType(cross_attn_probe, ca)

    # after you create `pipe = Wan2_2TI2VPipeline(...)` and before monkey-patching
    valid_layers = []
    for li, blk in enumerate(pipe.transformer.blocks):
        if (
            hasattr(blk, "cross_attn")
            and blk.cross_attn.__class__.__name__ == "WanCrossAttention"
        ):
            valid_layers.append(li)
    print(f"[info] cross-attn layers available: {valid_layers}")

    # keep only layers that actually exist
    req = sorted({int(x) for x in args.layers.split(",") if x.strip().isdigit()})
    missing = [L for L in req if L not in valid_layers]
    if missing:
        print(f"[warn] requested layers not present: {missing}")
    layers_to_log = [L for L in req if L in valid_layers]

    # -------------------------
    # Callback: record step id
    # -------------------------
    steps_seen = []

    def on_step_end(pipeline, i, t, kwargs):
        # update fallback step index so logging still works if transformer.current_steps isn’t set yet
        current_step["idx"] = int(i)
        steps_seen.append(int(i))
        return {"latents": kwargs["latents"]}

    # -------------------------
    # Generate (numpy video for easy overlays)
    # -------------------------
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
    vid_np = out.videos  # may be torch or numpy, any layout
    vid_uint8 = as_numpy_video_FHWC(
        vid_np if not isinstance(vid_np, list) else vid_np[0]
    )
    F_tgt, H, W, _ = vid_uint8.shape

    # latent query grid sizes
    t_patch, h_patch, w_patch = pipe.transformer.config.patch_size  # e.g. (1,2,2)
    vae_down = pipe.vae.config.spatial_compression_ratio
    Fp = (args.video_length - 1) // pipe.vae.config.temporal_compression_ratio + 1
    Hp = (args.height // vae_down) // h_patch
    Wp = (args.width // vae_down) // w_patch

    # which steps to visualize
    steps_sorted = sorted(attn_log.keys())
    if not steps_sorted:
        print("[warn] no attention captured. Check layers and spans.")
        return
    if step_filter is not None:
        steps_to_show = [s for s in steps_sorted if s in step_filter]
    elif step_mode == "last":
        steps_to_show = [steps_sorted[-1]]
    else:
        steps_to_show = steps_sorted

    # average or per-layer
    out_root = ensure_dir(args.save_path)
    for s in steps_to_show:
        lay_map = attn_log.get(s, {})
        # build token -> [B,Lq] (avg over layers if requested)
        per_token = []  # (name, Aflat, tag)
        for name in concept_list:
            mats, layers_used = [], []
            for L in layers_to_log:
                if L in lay_map and name in lay_map[L]:
                    mats.append(lay_map[L][name])
                    layers_used.append(L)
            if not mats:
                print(
                    f"[warn] no attention captured for '{name}' at step {s} across {layers_to_log}"
                )
                continue

            # one file per layer
            for A, L in zip(mats, layers_used):
                per_token.append((name, A, f"L{L}"))

            # optional averaged view
            if args.avg_layers and len(mats) > 1:
                Aavg = torch.stack(mats, dim=0).mean(dim=0)
                per_token.append((name, Aavg, "avgL"))

        # upsample -> overlay -> save
        step_dir = ensure_dir(os.path.join(out_root, f"step_{s:03d}"))
        for name, Aflat, tag in per_token:
            Aflat = Aflat[:, : Fp * Hp * Wp]
            A_up = marginals_to_frames(Aflat, Fp, Hp, Wp, F_tgt, H, W, norm="frame")
            heat = ensure_heat_FHW(A_up[0].numpy(), F_tgt, H, W)
            overlay = overlay_heatmap(vid_uint8, heat, alpha=args.alpha)
            path = os.path.join(step_dir, f"{name.replace('/','_')}_{tag}.mp4")
            save_np_FHWC_video(overlay, path, args.fps)

    # also save the plain result video
    # base_vid_path = os.path.join(out_root, "result.mp4")
    # base_t = (vid[0].transpose(3, 0, 1, 2)[None] * 255).clip(0, 255).astype(np.uint8)
    # base_t = torch.from_numpy(base_t)
    # save_videos_grid(base_t, base_vid_path, fps=args.fps)

    # save plain base result
    base_vid_path = os.path.join(out_root, "result.mp4")
    save_np_FHWC_video(vid_uint8, base_vid_path, args.fps)
    # manifest
    manifest = {
        "prompt": args.prompt,
        "image": args.image,
        "height": args.height,
        "width": args.width,
        "video_length": args.video_length,
        "fps": args.fps,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "shift": args.shift,
        "layers": layers_to_log,
        "concepts": concept_list,
        "steps_shown": steps_to_show,
        "avg_layers": args.avg_layers,
        "result_video": base_vid_path,
        "overlay_root": out_root,
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Overlays in: {out_root}")


if __name__ == "__main__":
    main()


# python examples/wan2.2/ti2v_token_frame_attn.py \
#   --prompt "a small orange elephant running in the field" \
#   --image /capstor/scratch/cscs/mhasan/VideoX-Fun-ours/test_assets/premium_photo-1669277330871-443462026e13.jpeg \
#   --height 704 --width 1280 --video-length 121 --fps 24 \
#   --num-steps 50 --guidance-scale 6.0 --sampler Flow_Unipc --shift 5 \
#   --layers 24,28,30 --avg-layers \
#   --concepts small,orange,elephant,running,field \
#   --steps last \
#   --save-path samples/wan2.2-token-frame-attn
