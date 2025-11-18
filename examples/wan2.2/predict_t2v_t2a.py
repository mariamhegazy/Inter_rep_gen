import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients

# --- NEW: prefer the T2A-aware pipeline if available; otherwise fall back to vanilla ---
PIPELINE_SUPPORTS_T2A = True

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

# ===========================
# Inference / system settings
# ===========================
GPU_memory_mode = "sequential_cpu_offload"  # ["model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"]
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False

# TeaCache
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

# CFG skip
cfg_skip_ratio = 0.0

# Riflex
enable_riflex = False
riflex_k = 6

# ===========================
# Model / sampler config
# ===========================
config_path = "config/wan2.2/wan_civitai_t2v.yaml"
model_name = "models/Wan2.2-T2V-A14B"

sampler_name = "Flow_Unipc"  # "Flow", "Flow_Unipc", "Flow_DPM++"
shift = 12  # used by Flow_Unipc / Flow_DPM++

# Optional checkpoints
transformer_path = None
transformer_high_path = None
vae_path = None

# Optional LoRA (low/high)
lora_path = "output_dir/wan2.2_14b_finetune_ultravideo_T2A_start_last_rank256_few_shot__long_5k_low_2/checkpoint-4000.safetensors"
lora_high_path = "output_dir/wan2.2_14b_finetune_ultravideo_T2A_start_last_rank256_few_shot__long_5k_high_2/checkpoint-4000.safetensors"

# ===========================
# Generation params
# ===========================
sample_size = [480, 832]
video_length = 81
fps = 16
weight_dtype = torch.bfloat16

# prompt = "The video captures a moment where a tree leaf turning from green to red. It starts with a completely green leaf and eventually change to red"
# prompt = "In a serene agricultural setting, a man dressed in a white polo shirt with a logo on the chest walks through a vibrant field of chili peppers. The scene opens with a wide shot of the field, showcasing rows of lush green plants dotted with bright red chilies. As the man progresses, the camera follows him from behind, transitioning to a medium shot that focuses on his interactions with the plants. He bends down occasionally, gently touching and examining the chilies, demonstrating a keen interest in their growth and quality. The background features dense foliage and a few trees, suggesting a well-maintained farm environment. The natural, overcast lighting enhances the colors of the plants and creates a soft, even illumination. The camera movements are smooth, with subtle pans and tilts, maintaining a steady pace that mirrors the man's deliberate and careful actions. The video's documentary style and naturalistic approach capture the essence of agricultural life, highlighting the man's expertise and dedication, and fostering a peaceful, grounded atmosphere that evokes a sense of connection with nature."
prompt = "In a quiet woodland edge in early autumn, a single maple leaf hangs from a thin branch. The scene opens with a tight macro shot, shallow depth of field isolating the leaf’s veins and serrated edges. The camera is on a tripod, static, with only a subtle breathing of the frame. Over the clip, the leaf’s color gradually shifts from deep green to rich red, starting at the edges and moving toward the center, with mottled yellow-orange phases visible along the veins. The background is a soft bokeh of distant trunks and foliage. Overcast natural lighting provides soft, even illumination that preserves texture and avoids glare. The pacing is calm and unhurried, consistent with a naturalistic documentary style and a single continuous shot without cuts."
# prompt = "A person is standing in a cozy kitchen, wearing a comfortable apron over a casual shirt and jeans. They are expertly chopping vegetables and stirring a pot on the stove, creating a warm and inviting atmosphere. Suddenly, they pause and turn to the pantry, where they begin to organize the shelves methodically, arranging cans and spices neatly. The kitchen is well-lit by warm, golden sunlight coming through the window behind them. The background shows clean countertops and other cooking utensils nearby, adding to the homey feel. The scene transitions smoothly from the active cooking to the focused organization, capturing the natural flow of daily life. Close-up shots of the person's face during both activities show their attentive and detail-oriented demeanor."
# prompt = "The race began, and the first runner from Team A quickly took off, leading the other teams. Everyone was intensely focused on his performance as he sprinted down the track. During the baton handoff, Team A nearly dropped the baton; however, the runner swiftly passed it to the second runner, who showed great determination. On the bend, the second runner chased hard and managed to overtake Team B, although the earlier mistake left Team A's lead only marginal. As the third runner from Team A nervously accelerated during the handoff, they managed to maintain the lead. However, Team C quickly caught up due to their excellent pace control. In the final sprint, the Team A runner accelerated in the second-to-last turn, gaining a burst of speed and widening the gap. Steadily running toward the finish line, the Team A runner crossed the line first, securing a thrilling victory. The intense atmosphere and close competition were captured in a dynamic, high-energy style, with smooth camera transitions between each handoff and sprint."
# prompt = "The camera orbits around Mount Fuji in a smooth, clockwise direction, capturing its majestic beauty from various angles. Mount Fuji stands tall and snow-capped against a clear blue sky, with its symmetrical cone shape standing out prominently. The surrounding landscape features lush green forests and serene lakes reflecting the mountain's silhouette. The background gradually transitions from daytime to a gentle sunset, with soft, warm hues lighting up the scene. The camera movement is fluid and dynamic, highlighting the mountain's intricate details and the tranquil atmosphere around it. Close-up to medium shot, with a gradual zoom-out to reveal the expansive landscape."
negative_prompt = (
    "overexposed, blurry, low quality, deformed hands, ugly, artifacts, static scene"
)
guidance_scale = 6.0
seed = 43
num_inference_steps = 50
lora_weight = 0.55
lora_high_weight = 0.55
save_path = "samples_test/wan-videos-t2v-startimg"

# --- NEW: T2A sparse-time knobs (mirrors your 5B script) ---
sparse_time_mode = "start_last"  # ["none","start","start_last","linspace_k","explicit_latent","explicit_pixel"]
sparse_apply_to = "both"  # ["latent","pixel","both"]
sparse_temporal_halo = 0  # extra pixel frames on each side when trimming
return_anchor_frames_only = True  # save only anchor frames (PNGs) instead of full MP4

# ===========================
# Build components
# ===========================
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

# Transformers (low/high noise)
transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(
        model_name,
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
transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(
        model_name,
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


# (optional) weight swaps
def _maybe_load(module, path):
    if path is None:
        return
    print(f"From checkpoint: {path}")
    if path.endswith("safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = module.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")


_maybe_load = _maybe_load
_maybe_load(transformer, transformer_path)
_maybe_load(transformer_2, transformer_high_path)

# VAE
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
}[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
).to(weight_dtype)
_maybe_load(vae, vae_path)

# Tokenizer + text encoder
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(
        model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    ),
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(
        model_name,
        config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
    ),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Scheduler
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
    config["scheduler_kwargs"]["shift"] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(
        Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])
    )
)

# Pipeline
pipeline = Wan2_2Pipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

# Multi-GPU helpers
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial

    transformer.enable_multi_gpus_inference()
    transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    for i in range(len(pipeline.transformer_2.blocks)):
        pipeline.transformer_2.blocks[i] = torch.compile(
            pipeline.transformer_2.blocks[i]
        )
    print("Add Compile")

# Offload / qfloat8 options
if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation"], device=device)
    replace_parameters_by_name(transformer_2, ["modulation"], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(
        transformer, exclude_module_name=["modulation"], device=device
    )
    convert_model_weight_to_float8(
        transformer_2, exclude_module_name=["modulation"], device=device
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(
        transformer, exclude_module_name=["modulation"], device=device
    )
    convert_model_weight_to_float8(
        transformer_2, exclude_module_name=["modulation"], device=device
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

# TeaCache / CFG-skip
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(
        f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps."
    )
    pipeline.transformer.enable_teacache(
        coefficients,
        num_inference_steps,
        teacache_threshold,
        num_skip_start_steps=num_skip_start_steps,
        offload=teacache_offload,
    )
    pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

# Seed
generator = torch.Generator(device=device).manual_seed(seed)

# LoRA merges
if lora_path is not None:
    pipeline = merge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )
    pipeline = merge_lora(
        pipeline,
        lora_high_path,
        lora_high_weight,
        device=device,
        dtype=weight_dtype,
        sub_transformer_name="transformer_2",
    )

with torch.no_grad():
    # Align requested video length to VAE compression
    video_length = (
        int(
            (video_length - 1)
            // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio
        )
        + 1
        if video_length != 1
        else 1
    )
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    # Optional Riflex
    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    # --- NEW: precompute & log anchors (just for user visibility) ---
    r = vae.config.temporal_compression_ratio  # e.g., 4
    T_pix = int(video_length)
    T_lat = (T_pix - 1) // r + 1
    if sparse_time_mode == "start":
        anchors_lat = [0]
    elif sparse_time_mode == "start_last":
        anchors_lat = [0, T_lat - 1] if T_lat > 1 else [0]
    else:
        anchors_lat = None  # for other modes the pipeline will compute
    if anchors_lat is not None:
        anchors_pix = [min(T_pix - 1, a * r) for a in anchors_lat]
        print(
            f"[sparse] r={r} T_pix={T_pix} T_lat={T_lat} anchors_lat={anchors_lat} anchors_pix={anchors_pix}"
        )
    # ensure we have two latent anchors when using start_last
    # r = vae.config.temporal_compression_ratio
    # if sparse_time_mode in ("start_last",) and video_length <= r:
    #     video_length = r + 1

    # # keep alignment to VAE ratio
    # video_length = int((video_length - 1) // r * r) + 1
    import inspect

    sig = inspect.signature(pipeline.__call__)
    t2a_kwargs = {}

    # if PIPELINE_SUPPORTS_T2A:
    #     t2a_kwargs.update(
    #         dict(
    #             sparse_time_mode=sparse_time_mode,
    #             sparse_time_arg=2,  # ignored by "start"/"start_last", required by API
    #             sparse_apply_to=sparse_apply_to,
    #             sparse_temporal_halo=sparse_temporal_halo,
    #             return_anchor_frames_only=return_anchor_frames_only,
    #         )
    #     )
    # if pipeline supports any sparse arg, we’ll pass the right ones
    if "sparse_time_mode" in sig.parameters:
        # common arg
        t2a_kwargs["sparse_time_mode"] = sparse_time_mode

        if "sparse_time_arg" in sig.parameters:
            # 5B/T2A-style names
            t2a_kwargs["sparse_time_arg"] = 2  # ignored for "start"/"start_last"
            if "sparse_apply_to" in sig.parameters:
                t2a_kwargs["sparse_apply_to"] = sparse_apply_to
            if "sparse_temporal_halo" in sig.parameters:
                t2a_kwargs["sparse_temporal_halo"] = sparse_temporal_halo
            if "return_anchor_frames_only" in sig.parameters:
                t2a_kwargs["return_anchor_frames_only"] = return_anchor_frames_only
        else:
            # 14B-style names
            if "sparse_k" in sig.parameters:
                t2a_kwargs["sparse_k"] = 2
            if "sparse_explicit" in sig.parameters:
                t2a_kwargs["sparse_explicit"] = ""
            if "sparse_return_anchors_only" in sig.parameters:
                t2a_kwargs["sparse_return_anchors_only"] = return_anchor_frames_only
            if "anchor_upsample" in sig.parameters:
                t2a_kwargs["anchor_upsample"] = "nearest"
    sample = pipeline(
        prompt,
        num_frames=video_length,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        boundary=boundary,
        shift=shift,
        **t2a_kwargs,
    ).videos

# Unmerge LoRA
if lora_path is not None:
    pipeline = unmerge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )
    pipeline = unmerge_lora(
        pipeline,
        lora_high_path,
        lora_high_weight,
        device=device,
        dtype=weight_dtype,
        sub_transformer_name="transformer_2",
    )


# ===========================
# Save
# ===========================
def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # make sure we have a torch tensor in [0,1]
    vid = sample
    if isinstance(vid, np.ndarray):
        vid = torch.from_numpy(vid)
    vid = vid.detach().cpu().float().clamp(0, 1)

    if vid.ndim != 5:
        raise RuntimeError(
            f"Unexpected output ndim={vid.ndim}, shape={tuple(vid.shape)}"
        )

    B, d1, d2, H, W = vid.shape
    prefix = str(len(os.listdir(save_path)) + 1).zfill(8)

    # If we asked the pipeline for anchors only, interpret as (B, K, C, H, W).
    if return_anchor_frames_only:
        K, C = d1, d2
        # (safety) if it looks like (B,C,T,H,W), swap
        if C not in (1, 3) and d1 in (1, 3):
            C, K = d1, d2
            vid = vid.permute(0, 2, 1, 3, 4).contiguous()  # now (B,K,C,H,W)

        if K == 1:
            img = vid[0, 0]  # (C,H,W)
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(save_path, prefix + ".png"))
        else:
            # save each anchor as its own PNG: prefix_anchor0.png, prefix_anchor1.png, ...
            for j in range(K):
                img = vid[0, j]  # (C,H,W)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    os.path.join(save_path, f"{prefix}_anchor{j}.png")
                )
    else:
        # We got a full clip, treat as (B,C,T,H,W) and save mp4
        # If tensor looks like (B,K,C,H,W), convert to (B,C,T,H,W)
        if d1 not in (1, 3) and d2 in (1, 3):
            vid = vid.permute(0, 2, 1, 3, 4).contiguous()
            _, C, T, _, _ = vid.shape
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(vid, video_path, fps=fps)


if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist

    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
