#!/usr/bin/env python3
import gc
import os
import sys

import numpy as np
import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from PIL import Image
from transformers import T5EncoderModel

# ---------- repo path ----------
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for pr in project_roots:
    if pr not in sys.path:
        sys.path.insert(0, pr)

# ---------- videox_fun ----------
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    T5Tokenizer,
)
from videox_fun.pipeline import CogVideoXFunInpaintPipeline, CogVideoXFunPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_grid


# ---------- helpers ----------
def load_ref_image_as_video(path, H, W, device, dtype=torch.float32):
    """[B=1, F=1, C=3, H, W] in [0,1]"""
    img = Image.open(path).convert("RGB")
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    img = img.resize((W, H), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,C)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C,H,W)
    t = t.unsqueeze(0).unsqueeze(0).contiguous()  # (1,1,C,H,W)
    return t.to(device=device, dtype=dtype)


def resolve_module_device(module, fallback_device):
    def to_device(x):
        if x is None:
            return None
        try:
            return torch.device(x)
        except (TypeError, ValueError):
            return None

    d = to_device(getattr(module, "device", None))
    if d is not None and d.type != "meta":
        return d
    try:
        pdev = next(module.parameters()).device
        d = to_device(pdev)
        if d is not None and d.type != "meta":
            return d
    except Exception:
        pass
    return to_device(fallback_device) or torch.device("cpu")


# ---------- config ----------
# GPU_memory_mode = "model_cpu_offload_and_qfloat8"
weight_dtype = torch.float32
text_encoder_dtype = torch.float32
vae_dtype = torch.float32
GPU_memory_mode = "model_cpu_offload"
ulysses_degree, ring_degree = 1, 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False

model_name = "models/CogVideoX1.5-5B"
sampler_name = "DDIM_Origin"

# Use your TI2V LoRA here
lora_path = "output_dir_cog/cog5b_rank128_inp/checkpoint-3500.safetensors"
# lora_path = None
lora_weight = 0.55

sample_size = [384, 672]  # [H, W]
video_length = 49  # V1.5: up to 85
fps = 8
ti2v_mode = "start"  # ["start","mid","last","random"]
# ref_path = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/2_dynamic_attr/state_0/0/1664_928_16:9_A_green_leaf_gpu0_img00.png"
ref_path = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/2_dynamic_attr/state_0/7/1664_928_16:9_Solid_ice_cream_gpu3_img00.png"

# dtypes — match your original working T2V script for CogVideoX
# weight_dtype       = torch.bfloat16
# text_encoder_dtype = torch.bfloat16
# vae_dtype          = torch.bfloat16

# prompt = "a tree leaf changing from green to red"
prompt = "an ice cream melting on a hot day"
negative_prompt = (
    "The video is not of a high quality, it has a low resolution. "
    "Watermark present in each frame. The background is solid. "
    "Strange body and strange trajectory. Distortion."
)
guidance_scale = 6.0
seed = 43
num_inference_steps = 50

save_path = "samples/cogvideox-fun-videos-ti2v"

# optional checkpoints
transformer_path = None
vae_path = None

# ---------- build ----------
device = set_multi_gpus_devices(ulysses_degree, ring_degree)

transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).to(weight_dtype)

if transformer_path:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(transformer_path)
    else:
        state = torch.load(transformer_path, map_location="cpu")
    state = state.get("state_dict", state)
    miss, unexp = transformer.load_state_dict(state, strict=False)
    print("### missing keys:", len(miss), "; ### unexpected keys:", len(unexp))

vae = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae").to(vae_dtype)

if vae_path:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(vae_path)
    else:
        state = torch.load(vae_path, map_location="cpu")
    state = state.get("state_dict", state)
    miss, unexp = vae.load_state_dict(state, strict=False)
    print("### missing keys:", len(miss), "; ### unexpected keys:", len(unexp))

tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=text_encoder_dtype
)

Chosen_Scheduler = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(model_name, subfolder="scheduler")

# Use the standard pipeline (supports TI2V with `video` + `ti2v_mode`)
pipeline = CogVideoXFunPipeline(
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    scheduler=scheduler,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial

    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(
            pipeline.transformer.transformer_blocks[i]
        )

# memory/offload
if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

# LoRA merge (TI2V finetune)
if lora_path:
    pipeline = merge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )

generator = torch.Generator(device=device).manual_seed(seed)

# ---------- CPU encode shim for TI2V conditioning ----------
# We DON'T precompute latents ourselves. We let the pipeline build the pixel tensor,
# then we override how it encodes that tensor.
_orig_encode_cond = pipeline._encode_conditioning_pixels


def _encode_conditioning_pixels_on_cpu(cond_pixels_lat: torch.Tensor):
    """
    cond_pixels_lat is expected as [B, C, F, H, W] in [-1,1] inside the pipeline.
    We force a CPU encode and return scaled latents on the model device.
    """
    # Move pixels to CPU float32
    cond_cpu = cond_pixels_lat.to("cpu", dtype=torch.float32, non_blocking=True)

    # Run a fresh CPU VAE (avoids CUDA conv3d)
    vae_cpu = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae").to(
        "cpu", dtype=torch.float32
    )
    with torch.autocast(device_type="cpu", enabled=False):
        lat = vae_cpu.encode(cond_cpu)[0].sample()  # [B, C_lat, F_lat, H_lat, W_lat]
    sf = float(getattr(vae_cpu.config, "scaling_factor", 1.0))
    lat = lat * sf

    # Cleanup helper VAE
    del vae_cpu
    gc.collect()

    # Return latents to the main model device/dtype
    target_device = resolve_module_device(pipeline.vae, device)
    target_dtype = getattr(pipeline.vae, "dtype", lat.dtype)
    return lat.to(device=target_device, dtype=target_dtype, non_blocking=True)


pipeline._encode_conditioning_pixels = _encode_conditioning_pixels_on_cpu

# ---------- run ----------
with torch.no_grad():
    # ensure frame count compatibility
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
    if (
        video_length != 1
        and getattr(transformer.config, "patch_size_t", None) is not None
        and latent_frames % transformer.config.patch_size_t != 0
    ):
        add_f = transformer.config.patch_size_t - (
            latent_frames % transformer.config.patch_size_t
        )
        video_length += add_f * vae.config.temporal_compression_ratio

    # reference frame as pixels in [0,1] (the pipeline will handle normalization)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found at {ref_path}")
    vae_dev = resolve_module_device(vae, device)
    ref_video = load_ref_image_as_video(
        ref_path,
        sample_size[0],
        sample_size[1],
        vae_dev,
        getattr(vae, "dtype", vae_dtype),
    )
    print("ref_video.shape =", tuple(ref_video.shape))  # (1,1,3,H,W)

    # sample with TI2V
    sample = pipeline(
        prompt=prompt,
        num_frames=video_length,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        video=ref_video,  # [B,1,3,H,W]
        ti2v_mode=ti2v_mode,
    ).videos  # [B, C, T, H, W] in [0,1]

    # Basic sanity print (no clamping/rescaling)
    s = sample.detach().to("cpu", torch.float32)
    print(
        "video stats -> min:",
        float(s.min()),
        "max:",
        float(s.max()),
        "mean:",
        float(s.mean()),
    )

# ---------- unmerge LoRA (optional) ----------
if lora_path:
    pipeline = unmerge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )


# ---------- save ----------
def save_results():
    os.makedirs(save_path, exist_ok=True)
    idx = len([p for p in os.listdir(save_path)]) + 1
    prefix = str(idx).zfill(8)
    if video_length == 1:
        out = os.path.join(save_path, prefix + ".png")
        img = (
            (sample[0, :, 0].transpose(0, 1).transpose(1, 2) * 255)
            .numpy()
            .astype(np.uint8)
        )
        Image.fromarray(img).save(out)
        print("Saved:", out)
    else:
        out = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, out, fps=fps)
        print("Saved:", out)


if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist

    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
