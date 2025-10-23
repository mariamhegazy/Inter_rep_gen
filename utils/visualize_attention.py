#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wan2.2 TI2V — Cross-attention *word* maps (spatiotemporal) with temporal alignment.

Outputs (in OUTPUT_DIR):
- attnmap_<word>.mp4        : per-frame heatmap over (F_p, H_p, W_p) showing queries attending to that word
- attnmap_<word>_sheet.png  : contact sheet PNG of the attention volume
- overlay_<word>.mp4        : attention overlaid on decoded frames (time-aligned)
- preview.mp4 / preview.png : generated video/image (for reference)

Fixes:
- Capture seq_lens/grid_sizes to slice Lq true and reshape to (F_p,H_p,W_p).
- Temporal resampling of attention volume to decoded frame timeline via VAE temporal ratio.
- Robust SentencePiece span matching for words ("elephant" etc.).
"""

import math
import os
import sys
import types
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

# ---- Project roots (same pattern you used) ----
this = os.path.abspath(__file__)
for r in [
    os.path.dirname(this),
    os.path.dirname(os.path.dirname(this)),
    os.path.dirname(os.path.dirname(os.path.dirname(this))),
]:
    if r not in sys.path:
        sys.path.insert(0, r)

import matplotlib.cm as cm
import torch.nn.functional as F

# ---- Wan2.2 imports you already use ----
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
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)

# =========================
# CONFIG (adjust to taste)
# =========================
MODEL_DIR = "models/Wan2.2-TI2V-5B"
CONFIG_PATH = "config/wan2.2/wan_civitai_5b.yaml"
OUTPUT_DIR = "samples/wan-attn-vis"

PROMPT = "a zebra running in the field"
WORDS_OF_INTEREST = ["zebra", "running", "field"]  # per-word maps will be saved

NEGATIVE_PROMPT = "overexposed, blurry, low quality, artifacts, static scene"
GUIDANCE_SCALE = 6.0
SEED = 43
WEIGHT_DTYPE = torch.bfloat16  # or torch.float16
VIDEO_LENGTH = 121  # will be snapped to VAE temporal compression
HEIGHT, WIDTH = 704, 1280
FPS = 24
NUM_STEPS = 50
SAMPLER_NAME = "Flow_Unipc"  # "Flow", "Flow_Unipc", "Flow_DPM++"
IMAGE_PATH = "/capstor/scratch/cscs/mhasan/VideoX-Fun-ours/test_assets/premium_photo-1669277330871-443462026e13.jpeg"

# GPU/offload knobs (keep consistent with your env)
GPU_MEMORY_MODE = "sequential_cpu_offload"
ULYSSES_DEGREE = 1
RING_DEGREE = 1
FSDP_DIT = False
FSDP_TEXT_ENCODER = True
COMPILE_DIT = False

ENABLE_TEACACHE = True
TEACACHE_THRESH = 0.10
TEACACHE_SKIP_STEPS = 5
TEACACHE_OFFLOAD = False

CFG_SKIP_RATIO = 0.0
ENABLE_RIFLEX = False
RIFLEX_K = 6

# --- viz controls ---
ATTN_VIZ = {
    "spatial_blur_sigma": 1.25,  # Gaussian blur in patch space; 0 = off
    "temporal_ema": 0.6,  # 0..1; smooth across frames; 0 = off
    "normalize": "frame",  # "global", "frame", or "percentile"
    "percentiles": (2.0, 98.0),  # for percentile normalization
    "overlay_alpha": 0.45,  # blend weight of heatmap on the frame
    "colormap": "magma",  # any mpl colormap name
    "upsample_to_video": True,  # resize attention to decoded frame size
    "contact_sheet_stride": max(1, FPS // 2),  # sparse frames on the sheet
    # temporal alignment knobs:
    "time_resample": "linear",  # "linear" or "repeat"
    "time_offset": "center",  # "center" or float in decoded-frame units (e.g., +0.5, -0.5)
}


# =========================
# Tracer capturing true Lq & grid + word maps
# =========================
class CrossAttnWordMapTracer:
    """
    Capture cross-attention and render per-word spatiotemporal maps:
      - From cross-attn, take softmax(QK^T / sqrt(d)) over text tokens, avg over heads.
      - Slice true queries (no padding) using seq_lens captured from WanAttentionBlock.forward.
      - Reshape Lq_true -> (F_p, H_p, W_p) using grid_sizes captured from WanAttentionBlock.forward.
      - Aggregate over layers & steps; save MP4 + contact sheet PNG per word.
      - Align attention timeline to decoded frames using VAE temporal compression ratio.
    """

    def __init__(
        self,
        tokenizer,
        text_len: int,
        save_dir: str,
        words_of_interest=None,
    ):
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.tokens: Optional[List[str]] = None
        self.valid_len: Optional[int] = None
        self.word2tok_idxs: Dict[str, List[int]] = {}
        self.words_of_interest = words_of_interest or []

        # storage
        self._per_layer_q2k: Dict[int, torch.Tensor] = {}
        self._per_step_layers: Dict[int, Dict[int, torch.Tensor]] = {}
        self._current_step = 0
        self._seen_block0_this_step = False
        self._block0_module = None
        self._orig_forwards = {}

        # dynamic info captured from WanAttentionBlock.forward each call
        self._current_seq_lens: Optional[torch.Tensor] = None  # [B]
        self._current_grid_sizes: Optional[torch.Tensor] = None  # [B,3]
        self._vae_tcomp_ratio: Optional[int] = None  # set from main()

    # ----------------- Tokenization helpers (SP spans) -----------------
    def _normalize_sp_token(self, tok: str) -> str:
        return tok.replace("▁", "").lower()

    def _find_word_token_spans(self, tokens: List[str], word: str) -> List[List[int]]:
        """Find contiguous token spans whose concatenation equals `word` (SentencePiece)."""
        w = word.lower()
        norm = [self._normalize_sp_token(t) for t in tokens]
        spans = []
        n = len(norm)
        i = 0
        while i < n:
            # start only if compatible prefix
            if not w.startswith(norm[i]) and norm[i] != w and norm[i] not in w:
                i += 1
                continue
            j = i
            acc = ""
            while j < n and len(acc) <= len(w):
                acc += norm[j]
                if acc == w:
                    spans.append(list(range(i, j + 1)))
                    break
                if not w.startswith(acc):
                    break
                j += 1
            i += 1
        return spans

    def set_prompt_tokens(self, prompt: str):
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        self.tokens = self.tokenizer.convert_ids_to_tokens(ids)
        self.valid_len = len(self.tokens)

        # debug print so you can see the split
        tok_print = [f"{i}:{t}" for i, t in enumerate(self.tokens)]
        print("[tokenization]", " | ".join(tok_print))

        self.word2tok_idxs = {}
        for w in self.words_of_interest:
            spans = self._find_word_token_spans(self.tokens, w)
            if spans:
                flat = sorted({idx for s in spans for idx in s})
                self.word2tok_idxs[w] = flat
            else:
                # fallback substring scan (handles punctuation)
                low_toks = [self._normalize_sp_token(t) for t in self.tokens]
                fallback = [
                    i
                    for i, lt in enumerate(low_toks)
                    if (lt in w.lower()) or (w.lower() in lt)
                ]
                self.word2tok_idxs[w] = fallback

    # ----------------- Patching & capture -----------------
    def _maybe_new_step(self, module, layer_idx: int):
        if self._block0_module is None and layer_idx == 0:
            self._block0_module = module
        if module is self._block0_module:
            if self._seen_block0_this_step:
                return
            self._seen_block0_this_step = True
            if self._current_step not in self._per_step_layers:
                self._per_step_layers[self._current_step] = {}

    @staticmethod
    def _compute_probs(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q: [B, Lq, H, D], k: [B, Lk, H, D]
        d = q.size(-1)
        qh = q.permute(0, 2, 1, 3)  # [B,H,Lq,D]
        kh = k.permute(0, 2, 3, 1)  # [B,H,D,Lk]
        scores = torch.matmul(qh.float(), kh.float()) / math.sqrt(d)
        probs = torch.softmax(scores, dim=-1)  # [B,H,Lq,Lk]
        probs = probs.mean(dim=1)  # avg heads -> [B,Lq,Lk]
        return probs

    def attach(self, transformer):
        """Patch WanAttentionBlock.forward to cache seq_lens/grid_sizes,
        and patch WanAttentionBlock.cross_attn.forward to capture cross-attn probs."""
        # 1) patch block.forward
        for layer_idx, block in enumerate(transformer.blocks):
            orig_block_forward = block.forward
            self._orig_forwards[(block, "forward")] = orig_block_forward

            def make_block_forward_patched(_layer_idx, _orig):
                def patched_block_forward(
                    self_block,
                    x,
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                    dtype=torch.bfloat16,
                    t=0,
                ):
                    self._current_seq_lens = seq_lens.detach().cpu()
                    self._current_grid_sizes = grid_sizes.detach().cpu()
                    return _orig(
                        x,
                        e,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                    )

                return patched_block_forward

            block.forward = types.MethodType(
                make_block_forward_patched(layer_idx, orig_block_forward), block
            )

        # 2) patch cross_attn.forward
        for layer_idx, block in enumerate(transformer.blocks):
            mod = block.cross_attn
            orig = mod.forward
            self._orig_forwards[(mod, "forward")] = orig

            def make_cross_patched(_layer_idx, _orig):
                def patched_cross(
                    self_mod, x, context, context_lens, dtype=torch.bfloat16, t=0
                ):
                    b = x.size(0)
                    n = self_mod.num_heads
                    d = self_mod.head_dim
                    q = self_mod.norm_q(self_mod.q(x.to(dtype))).view(b, -1, n, d)
                    k = self_mod.norm_k(self_mod.k(context.to(dtype))).view(b, -1, n, d)

                    self._maybe_new_step(self_mod, _layer_idx)
                    probs = self._compute_probs(q, k)  # [B,Lq_pad,Lk]
                    if self.valid_len is not None:
                        probs = probs[:, :, : self.valid_len]
                    if self._current_seq_lens is not None:
                        true_Lq = int(self._current_seq_lens[0].item())
                        probs = probs[:, :true_Lq, :]  # [B, Lq_true, Lk]

                    probs_cpu = probs.detach().cpu()

                    if _layer_idx not in self._per_layer_q2k:
                        self._per_layer_q2k[_layer_idx] = probs_cpu
                    else:
                        self._per_layer_q2k[_layer_idx] += probs_cpu

                    if self._current_step not in self._per_step_layers:
                        self._per_step_layers[self._current_step] = {}
                    if _layer_idx not in self._per_step_layers[self._current_step]:
                        self._per_step_layers[self._current_step][
                            _layer_idx
                        ] = probs_cpu
                    else:
                        self._per_step_layers[self._current_step][
                            _layer_idx
                        ] += probs_cpu

                    return _orig(x, context, context_lens, dtype, t)

                return patched_cross

            block.cross_attn.forward = types.MethodType(
                make_cross_patched(layer_idx, orig), block.cross_attn
            )

        # 3) wrap transformer.forward to tick step
        orig_tf_forward = transformer.forward
        self._orig_forwards[(transformer, "forward")] = orig_tf_forward

        def tf_forward_with_step(
            self_tf,
            x,
            t,
            context,
            seq_len,
            clip_fea=None,
            y=None,
            y_camera=None,
            full_ref=None,
            subject_ref=None,
            cond_flag=True,
        ):
            self._seen_block0_this_step = False
            out = orig_tf_forward(
                x=x,
                t=t,
                context=context,
                seq_len=seq_len,
                clip_fea=clip_fea,
                y=y,
                y_camera=y_camera,
                full_ref=full_ref,
                subject_ref=subject_ref,
                cond_flag=cond_flag,
            )
            self._current_step += 1
            return out

        transformer.forward = types.MethodType(tf_forward_with_step, transformer)

    def detach(self):
        for (obj, name), orig in list(self._orig_forwards.items()):
            setattr(obj, name, orig)
            del self._orig_forwards[(obj, name)]

    # ---------- Rendering & utilities ----------
    @staticmethod
    def _normalize(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        m, M = float(a.min()), float(a.max())
        if M - m < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - m) / (M - m + 1e-8)

    def _make_contact_sheet(self, vol, out_png, every=1, max_cols=8):
        F = vol.shape[0]
        idxs = list(range(0, F, every)) or [0]
        imgs = []
        for i in idxs:
            sl = self._normalize(vol[i])
            sl = (sl * 255).astype(np.uint8)
            sl = np.repeat(np.repeat(sl, 8, axis=0), 8, axis=1)
            imgs.append(np.stack([sl] * 3, axis=-1))
        rows = int(np.ceil(len(imgs) / max_cols))
        cols = min(len(imgs), max_cols)
        h, w, _ = imgs[0].shape
        sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for n, im in enumerate(imgs):
            r, c = n // cols, n % cols
            sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
        plt.imsave(out_png, sheet)

    def _write_video(self, vol, out_mp4, fps=8):
        frames = []
        for i in range(vol.shape[0]):
            sl = self._normalize(vol[i])
            sl = (sl * 255).astype(np.uint8)
            sl = np.repeat(np.repeat(sl, 8, axis=0), 8, axis=1)
            frames.append(np.stack([sl] * 3, axis=-1))
        with iio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr)

    def _aggregate_q2k_and_grid(self):
        if not self._per_layer_q2k:
            raise RuntimeError("No cross-attention captured.")
        layers = sorted(self._per_layer_q2k.keys())
        agg = sum(self._per_layer_q2k[li] for li in layers) / max(
            1, len(layers)
        )  # [B,Lq_true,Lk]
        agg = agg.mean(dim=0).numpy()  # [Lq_true, Lk]
        if self._current_grid_sizes is None:
            raise RuntimeError("grid_sizes not captured from WanAttentionBlock.forward")
        gs = self._current_grid_sizes[0].tolist()  # [F_p, H_p, W_p]
        F_p, H_p, W_p = int(gs[0]), int(gs[1]), int(gs[2])
        return agg, F_p, H_p, W_p

    def _gaussian_kernel(self, sigma: float, device):
        if sigma <= 0:
            return None
        k = int(max(3, 2 * int(3 * sigma) + 1))
        x = torch.arange(k, device=device) - (k // 2)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        return g

    def _blur_spatial(self, vol_np: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return vol_np
        v = torch.from_numpy(vol_np).unsqueeze(1)  # [F,1,H,W]
        g = self._gaussian_kernel(sigma, v.device)
        if g is None:
            return vol_np
        g1 = g.view(1, 1, 1, -1)
        g2 = g.view(1, 1, -1, 1)
        v = F.conv2d(v, g1, padding=(0, g1.size(-1) // 2), groups=1)
        v = F.conv2d(v, g2, padding=(g2.size(-2) // 2, 0), groups=1)
        return v.squeeze(1).cpu().numpy()

    def _ema_temporal(self, vol_np: np.ndarray, beta: float) -> np.ndarray:
        if beta <= 0:
            return vol_np
        out = vol_np.copy()
        for t in range(1, out.shape[0]):
            out[t] = beta * out[t - 1] + (1.0 - beta) * out[t]
        return out

    def _normalize_vol(
        self, vol: np.ndarray, mode="percentile", p=(2, 98)
    ) -> np.ndarray:
        if mode == "frame":
            v = vol.copy()
            for i in range(v.shape[0]):
                m, M = v[i].min(), v[i].max()
                v[i] = 0 if M - m < 1e-8 else (v[i] - m) / (M - m)
            return v
        if mode == "percentile":
            lo, hi = np.percentile(vol, p[0]), np.percentile(vol, p[1])
            return np.clip((vol - lo) / max(hi - lo, 1e-8), 0, 1)
        m, M = vol.min(), vol.max()
        return np.zeros_like(vol) if M - m < 1e-8 else (vol - m) / (M - m)

    def _colorize(self, gray_01: np.ndarray, cmap_name="magma") -> np.ndarray:
        cmap = cm.get_cmap(cmap_name)
        rgb = cmap(np.clip(gray_01, 0, 1))[..., :3]
        return (rgb * 255).astype(np.uint8)

    def _upsample_to_frame(self, attn_2d, target_hw) -> np.ndarray:
        th, tw = target_hw
        from PIL import Image

        im = Image.fromarray((attn_2d * 255).astype(np.uint8), mode="L")
        im = im.resize((tw, th), resample=Image.BICUBIC)
        return np.asarray(im).astype(np.float32) / 255.0

    # --------- Temporal alignment (latent timeline -> decoded frames) ---------
    def _temporal_resample_to_decoded(
        self, vol: np.ndarray, F_dec: int, r: int, mode="linear", offset="center"
    ) -> np.ndarray:
        """
        vol: [F_p, H_p, W_p] on latent timeline
        F_dec: #decoded frames
        r: VAE temporal compression ratio
        mode: "repeat" (nearest) or "linear"
        offset: "center" -> center each latent slice at p*r + (r-1)/2,
                or float offset in decoded-frame units
        """
        F_p = vol.shape[0]
        if F_p <= 1:
            return np.repeat(vol, max(1, F_dec), axis=0)[:F_dec]
        if F_dec == F_p:
            return vol

        # source positions in decoded-frame units
        if offset == "center":
            t_src = np.arange(F_p) * r + (r - 1) / 2.0
        else:
            t_src = np.arange(F_p) * r + float(offset)
        t_src = np.clip(t_src, 0.0, float(F_dec - 1))
        t_tgt = np.arange(F_dec).astype(np.float32)

        if mode == "repeat":
            idx = np.clip(np.rint(t_tgt / r).astype(int), 0, F_p - 1)
            return vol[idx]
        else:
            right = np.clip(np.searchsorted(t_src, t_tgt, side="left"), 0, F_p - 1)
            left = np.clip(right - 1, 0, F_p - 1)
            denom = t_src[right] - t_src[left] + 1e-8
            w = (t_tgt - t_src[left]) / denom
            w = w[:, None, None]
            return (1.0 - w) * vol[left] + w * vol[right]

    def _write_overlay_video(
        self,
        vol_01: np.ndarray,
        video_tensor: torch.Tensor,
        alpha=0.45,
        cmap="magma",
        out_mp4="overlay.mp4",
        fps=8,
    ):
        """
        vol_01: [F_dec, H_p, W_p] aligned to decoded timeline (0..1)
        video_tensor: [C, F_dec, H, W] in 0..1 (decoded sample for one item)
        """
        C, F, H, W = video_tensor.shape
        frames = []
        for t in range(min(F, vol_01.shape[0])):
            base = (video_tensor[:, t].permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            a2d = self._upsample_to_frame(vol_01[t], (H, W))
            heat = self._colorize(a2d, cmap)
            blended = (
                (1 - alpha) * base.astype(np.float32) + alpha * heat.astype(np.float32)
            ).astype(np.uint8)
            frames.append(blended)
        with iio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr)

    # ----------------- Public rendering -----------------
    def render_word_maps(self, fps=8, attn_viz=None, sample_video=None):
        if attn_viz is None:
            attn_viz = {}
        agg, F_p, H_p, W_p = self._aggregate_q2k_and_grid()
        Lq_true, Lk = agg.shape

        def to_volume(vec_1d: np.ndarray) -> np.ndarray:
            if Lq_true == F_p * H_p * W_p:
                return vec_1d.reshape(F_p, H_p, W_p)
            HW = Lq_true // max(1, F_p)
            W_eff = max(1, int(round(math.sqrt(HW))))
            H_eff = max(1, HW // W_eff)
            return vec_1d.reshape(F_p, H_eff, W_eff)

        for word, tok_idxs in self.word2tok_idxs.items():
            if not tok_idxs:
                print(f"[warn] No tokens for '{word}'. Skipping.")
                continue
            tok_idxs = [i for i in tok_idxs if i < (self.valid_len or Lk)]
            if not tok_idxs:
                print(f"[warn] '{word}' tokens beyond valid_len. Skipping.")
                continue

            col = agg[:, tok_idxs].mean(axis=1)  # [Lq_true]
            vol = to_volume(col)  # [F_p,H_p,W_p]

            # smoothing & normalization
            if attn_viz.get("spatial_blur_sigma", 0) > 0:
                vol = self._blur_spatial(vol, attn_viz["spatial_blur_sigma"])
            if attn_viz.get("temporal_ema", 0) > 0:
                vol = self._ema_temporal(vol, attn_viz["temporal_ema"])
            vol = self._normalize_vol(
                vol,
                mode=attn_viz.get("normalize", "percentile"),
                p=attn_viz.get("percentiles", (2, 98)),
            )

            # save base latent-timeline heatmap
            safe = word.replace("/", "_")
            mp4 = os.path.join(self.save_dir, f"attnmap_{safe}.mp4")
            png = os.path.join(self.save_dir, f"attnmap_{safe}_sheet.png")
            self._write_video(vol, mp4, fps=fps)
            self._make_contact_sheet(
                vol, png, every=attn_viz.get("contact_sheet_stride", max(1, fps // 2))
            )
            print(f"[ok] Saved latent attention map for '{word}' → {mp4}, {png}")

            # overlay on decoded frames with temporal alignment
            if sample_video is not None and attn_viz.get("upsample_to_video", True):
                C, F_dec, H_dec, W_dec = sample_video[0].shape
                # Get VAE ratio
                r = (
                    int(self._vae_tcomp_ratio)
                    if self._vae_tcomp_ratio
                    else max(1, int(round(F_dec / max(1, vol.shape[0]))))
                )
                vol_aligned = self._temporal_resample_to_decoded(
                    vol,
                    F_dec,
                    r,
                    mode=attn_viz.get("time_resample", "linear"),
                    offset=attn_viz.get("time_offset", "center"),
                )
                ov = os.path.join(self.save_dir, f"overlay_{safe}.mp4")
                self._write_overlay_video(
                    vol_01=vol_aligned,
                    video_tensor=sample_video[0],
                    alpha=attn_viz.get("overlay_alpha", 0.45),
                    cmap=attn_viz.get("colormap", "magma"),
                    out_mp4=ov,
                    fps=fps,
                )
                print(f"[ok] Saved time-aligned overlay for '{word}' → {ov}")

    def render_word_maps_by_step(
        self, word: str, out_dir: str, fps=8, attn_viz=None, sample_video=None
    ):
        if attn_viz is None:
            attn_viz = {}
        steps = sorted(self._per_step_layers.keys())
        agg, F_p, H_p, W_p = self._aggregate_q2k_and_grid()  # for F_p,H_p,W_p
        valid_len = self.valid_len or agg.shape[1]
        # span-based indices if available
        spans = self._find_word_token_spans(self.tokens, word)
        tok_idxs = sorted({i for s in spans for i in s}) if spans else []
        if not tok_idxs:
            tok_idxs = [
                i for i in range(valid_len) if i in (self.word2tok_idxs.get(word, []))
            ]
        if not tok_idxs:
            print(f"[warn] no tokens for '{word}' at step render")
            return

        r = int(self._vae_tcomp_ratio) if self._vae_tcomp_ratio else None
        os.makedirs(out_dir, exist_ok=True)
        safe = word.replace("/", "_")
        for step in steps:
            layers = sorted(self._per_step_layers[step].keys())
            q2k = sum(self._per_step_layers[step][li] for li in layers) / max(
                1, len(layers)
            )  # [B,Lq_true,Lk]
            q2k = q2k.mean(dim=0).numpy()  # [Lq_true,Lk]
            col = q2k[:, tok_idxs].mean(axis=1)
            vol = col.reshape(F_p, H_p, W_p)

            if attn_viz.get("spatial_blur_sigma", 0) > 0:
                vol = self._blur_spatial(vol, attn_viz["spatial_blur_sigma"])
            if attn_viz.get("temporal_ema", 0) > 0:
                vol = self._ema_temporal(vol, attn_viz["temporal_ema"])
            vol = self._normalize_vol(
                vol,
                mode=attn_viz.get("normalize", "percentile"),
                p=attn_viz.get("percentiles", (2, 98)),
            )

            if sample_video is not None and attn_viz.get("upsample_to_video", True):
                C, F_dec, _, _ = sample_video[0].shape
                rr = (
                    r
                    if r is not None
                    else max(1, int(round(F_dec / max(1, vol.shape[0]))))
                )
                vol_aligned = self._temporal_resample_to_decoded(
                    vol,
                    F_dec,
                    rr,
                    mode=attn_viz.get("time_resample", "linear"),
                    offset=attn_viz.get("time_offset", "center"),
                )
                out_mp4 = os.path.join(out_dir, f"overlay_{safe}_step{step:03d}.mp4")
                self._write_overlay_video(
                    vol_01=vol_aligned,
                    video_tensor=sample_video[0],
                    alpha=attn_viz.get("overlay_alpha", 0.45),
                    cmap=attn_viz.get("colormap", "magma"),
                    out_mp4=out_mp4,
                    fps=fps,
                )


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # device & config
    device = set_multi_gpus_devices(ULYSSES_DEGREE, RING_DEGREE)
    config = OmegaConf.load(CONFIG_PATH)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    # --- Transformer(s)
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            MODEL_DIR,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
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
                MODEL_DIR,
                config["transformer_additional_kwargs"].get(
                    "transformer_high_noise_model_subpath", "transformer"
                ),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=WEIGHT_DTYPE,
        )

    # --- VAE
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(MODEL_DIR, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(WEIGHT_DTYPE)

    # --- Tokenizer & Text Encoder
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            MODEL_DIR,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            MODEL_DIR,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    )

    # --- Scheduler
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[SAMPLER_NAME]
    sched_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    if SAMPLER_NAME in ("Flow_Unipc", "Flow_DPM++"):
        sched_kwargs["shift"] = 1
    scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, sched_kwargs))

    # --- Pipeline
    pipe = Wan2_2TI2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Multi-GPU / offload settings (mirror your script)
    if ULYSSES_DEGREE > 1 or RING_DEGREE > 1:
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if FSDP_DIT:
            from functools import partial

            shard_fn = partial(shard_model, device_id=device, param_dtype=WEIGHT_DTYPE)
            pipe.transformer = shard_fn(pipe.transformer)
            if transformer_2 is not None:
                pipe.transformer_2 = shard_fn(pipe.transformer_2)
        if FSDP_TEXT_ENCODER:
            from functools import partial

            shard_fn = partial(shard_model, device_id=device, param_dtype=WEIGHT_DTYPE)
            pipe.text_encoder = shard_fn(pipe.text_encoder)

    if COMPILE_DIT:
        for i in range(len(pipe.transformer.blocks)):
            pipe.transformer.blocks[i] = torch.compile(pipe.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipe.transformer_2.blocks)):
                pipe.transformer_2.blocks[i] = torch.compile(
                    pipe.transformer_2.blocks[i]
                )

    if GPU_MEMORY_MODE == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipe.enable_sequential_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(
            transformer, exclude_module_name=["modulation"], device=device
        )
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        if transformer_2 is not None:
            convert_model_weight_to_float8(
                transformer_2, exclude_module_name=["modulation"], device=device
            )
            convert_weight_dtype_wrapper(transformer_2, WEIGHT_DTYPE)
        pipe.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    elif GPU_MEMORY_MODE == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(
            transformer, exclude_module_name=["modulation"], device=device
        )
        convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
        if transformer_2 is not None:
            convert_model_weight_to_float8(
                transformer_2, exclude_module_name=["modulation"], device=device
            )
            convert_weight_dtype_wrapper(transformer_2, WEIGHT_DTYPE)
        pipe.to(device=device)
    else:
        pipe.to(device=device)

    # TeaCache / cfg skip / riflex (optional)
    if ENABLE_TEACACHE:
        coeffs = get_teacache_coefficients(MODEL_DIR)
        if coeffs is not None:
            pipe.transformer.enable_teacache(
                coeffs,
                NUM_STEPS,
                TEACACHE_THRESH,
                num_skip_start_steps=TEACACHE_SKIP_STEPS,
                offload=TEACACHE_OFFLOAD,
            )
            if transformer_2 is not None:
                transformer_2.share_teacache(transformer=pipe.transformer)

    if CFG_SKIP_RATIO and CFG_SKIP_RATIO > 0:
        pipe.transformer.enable_cfg_skip(CFG_SKIP_RATIO, NUM_STEPS)
        if transformer_2 is not None:
            transformer_2.share_cfg_skip(transformer=pipe.transformer)

    # Prepare TI2V latent
    with torch.no_grad():
        video_length = (
            int(
                (VIDEO_LENGTH - 1)
                // vae.config.temporal_compression_ratio
                * vae.config.temporal_compression_ratio
            )
            + 1
            if VIDEO_LENGTH != 1
            else 1
        )
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if ENABLE_RIFLEX:
            pipe.transformer.enable_riflex(k=RIFLEX_K, L_test=latent_frames)
            if transformer_2 is not None:
                pipe.transformer_2.enable_riflex(k=RIFLEX_K, L_test=latent_frames)

        input_video, input_video_mask, _ = get_image_to_video_latent(
            IMAGE_PATH, None, video_length=video_length, sample_size=[HEIGHT, WIDTH]
        )

    # ============ Attach tracer ============
    tracer = CrossAttnWordMapTracer(
        tokenizer=tokenizer,
        text_len=pipe.transformer.text_len,
        save_dir=OUTPUT_DIR,
        words_of_interest=WORDS_OF_INTEREST,
    )
    tracer.set_prompt_tokens(PROMPT)
    tracer.attach(pipe.transformer)
    if transformer_2 is not None:
        tracer.attach(pipe.transformer_2)

    # pass VAE temporal compression ratio for alignment
    tracer._vae_tcomp_ratio = int(vae.config.temporal_compression_ratio)

    # ============ Run sampling ============
    generator = torch.Generator(device=device).manual_seed(SEED)
    with torch.no_grad():
        sample = pipe(
            PROMPT,
            num_frames=video_length,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
        ).videos  # [B, C, F, H, W], in [0,1]

    # Detach tracer and render word maps
    tracer.detach()
    tracer.render_word_maps(fps=FPS, attn_viz=ATTN_VIZ, sample_video=sample)
    # tracer.render_word_maps_by_step(
    #     "zebra",
    #     os.path.join(OUTPUT_DIR, "by_step"),
    #     fps=FPS,
    #     attn_viz=ATTN_VIZ,
    #     sample_video=sample,
    # )

    # Save the generated output next to the maps
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if video_length == 1:
        fn = os.path.join(OUTPUT_DIR, "preview.png")
        image = sample[0, :, 0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((image * 255).astype(np.uint8)).save(fn)
    else:
        fn = os.path.join(OUTPUT_DIR, "preview.mp4")
        save_videos_grid(sample, fn, fps=FPS)

    print(f"[OK] Saved attention word-maps and media to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
