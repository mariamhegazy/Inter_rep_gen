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


import math
import os

import matplotlib.pyplot as plt
import torch

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


def _reshape_to_fhw(vec, grid):
    # vec: [Lq], grid: [3] = (F, H, W)
    F, H, W = [int(x) for x in grid.tolist()]
    tensor = vec[: F * H * W].reshape(F, H, W)
    return tensor.float().cpu().numpy()


def save_zebra_heatmaps(attn_module, out_dir="debug_attn", prefix="run"):
    """
    attn_module: the *last* WanCrossAttention you want to inspect
                 (e.g., pipeline.transformer.blocks[-1].cross_attn)
    """
    os.makedirs(out_dir, exist_ok=True)
    info = getattr(attn_module, "_debug_last", None)
    if info is None or info["zebra_mass"] is None or info["grid"] is None:
        print("No debug info found on attn module.")
        return

    z = info["zebra_mass"][0]  # [Lq] for batch item 0
    grid = info["grid"][0]  # [3] (F,H,W) for sample 0
    fhw = _reshape_to_fhw(z, grid)  # [F,H,W]

    # aggregate over time to see spatial layout; and over space to see per-frame curve
    spatial = fhw.mean(0)  # [H,W]
    per_frame = fhw.mean(axis=(1, 2))  # [F]

    # optional: overlay the gate if present (after steering)
    gate = info.get("gate", None)
    if gate is not None:
        g_fhw = _reshape_to_fhw(gate[0], grid)  # [F,H,W]
        g_spatial = g_fhw.mean(0)  # [H,W]
    else:
        g_spatial = None

    # Plot spatial maps
    plt.figure()
    plt.title("Zebra attention mass (spatial avg over frames)")
    plt.imshow(spatial, cmap="inferno")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_zebra_spatial.png"))
    plt.close()

    if g_spatial is not None:
        plt.figure()
        plt.title(
            "Applied gate g (spatial avg over frames) — lower = stronger suppression"
        )
        plt.imshow(g_spatial, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_gate_spatial.png"))
        plt.close()

    # Plot per-frame curve
    plt.figure()
    plt.title("Zebra attention mass per frame (mean over HxW)")
    plt.plot(per_frame)
    plt.xlabel("frame")
    plt.ylabel("mass")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_zebra_per_frame.png"))
    plt.close()

    print("wrote:", out_dir)


# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
#
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
#
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree = 1
ring_degree = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit = False
fsdp_text_encoder = True
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit = False

# TeaCache config
enable_teacache = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process,
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio = 0

# Riflex config
enable_riflex = False
# Index of intrinsic frequency
riflex_k = 6

# Config and model path
config_path = "config/wan2.2/wan_civitai_5b.yaml"
# model path
model_name = "models/Wan2.2-TI2V-5B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics.
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only transformer_path is used.
transformer_path = None
transformer_high_path = None
vae_path = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only lora_path is used.
lora_path = None
lora_high_path = None

# Other params
sample_size = [704, 1280]
video_length = 121
fps = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start = "/capstor/scratch/cscs/mhasan/VideoX-Fun-ours/test_assets/premium_photo-1669277330871-443462026e13.jpeg"

# prompts
prompt = "a zebra running in the field"
negative_prompt = (
    "overexposed, blurry, low quality, deformed hands, ugly, artifacts, static scene"
)
guidance_scale = 6.0
seed = 43
num_inference_steps = 50
# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight = 0.55
lora_high_weight = 0.55
save_path = "samples/wan-videos-ti2v-adapt"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

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
if (
    config["transformer_additional_kwargs"].get(
        "transformer_combination_type", "single"
    )
    == "moe"
):
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
else:
    transformer_2 = None

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open

        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = (
            state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        )

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
}[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open

        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(
        model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    ),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(
        model_name,
        config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
    ),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config["scheduler_kwargs"]["shift"] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(
        Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])
    )
)

# Get Pipeline
pipeline = Wan2_2TI2VPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial

    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(
                pipeline.transformer_2.blocks[i]
            )
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(
        transformer,
        [
            "modulation",
        ],
        device=device,
    )
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(
            transformer_2,
            [
                "modulation",
            ],
            device=device,
        )
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=[
            "modulation",
        ],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(
            transformer_2,
            exclude_module_name=[
                "modulation",
            ],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=[
            "modulation",
        ],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(
            transformer_2,
            exclude_module_name=[
                "modulation",
            ],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

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
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )
    if transformer_2 is not None:
        pipeline = merge_lora(
            pipeline,
            lora_high_path,
            lora_high_weight,
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )

with torch.no_grad():
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

    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    if validation_image_start is not None:
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            validation_image_start,
            None,
            video_length=video_length,
            sample_size=sample_size,
        )
    else:
        input_video, input_video_mask, clip_image = None, None, None

    tokens = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    ids = tokens.input_ids[0].tolist()

    # find all positions belonging to the word "zebra"
    # (handles BPE subwords like "ze", "##bra" etc.; adapt as needed to your tokenizer)
    zebra_positions = [
        i
        for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids))
        if "zebra" in tok.lower()
    ]

    # 2) enable steering on the transformer
    # pipeline.transformer.enable_text_spatial_steer(
    #     token_ids=zebra_positions,
    #     beta=0.6,  # suppression strength (try 0.4 ~ 0.7)
    #     ema=0.8,  # smoother memory of where zebra tends to sit
    #     topk=0.10,  # suppress top 10% zebra-dominated query tokens each layer call
    # )

    # pipeline.transformer.enable_text_framewise_crossattn(
    #     start=0.2, end=0.3, mode="linear"
    # )

    # # --- Temporal fade for the image-conditioning mask ---
    # if input_video_mask is not None:
    #     # input_video_mask: (1, 1, T, H, W) with 1's where the copied image dominates
    #     T = input_video_mask.shape[2]

    #     # Linear decay (1.0 at frame 0 → 0.0 at frame T-1).
    #     # Feel free to try 'cosine' or 'exp' schedules.
    #     decay = torch.linspace(
    #         1.0, 0.0, T, device=input_video_mask.device, dtype=input_video_mask.dtype
    #     )

    #     # Optional: keep first frame slightly looser than 1.0 if you want text to already nudge it
    #     # decay[0] = 0.8

    #     input_video_mask = input_video_mask * decay.view(1, 1, T, 1, 1)

    # def make_framewise_cfg_wrapper(pipe, base_s=3.0, g0=1.0, g1=7.0):
    #     """
    #     Keep Wan2.2 pipeline's CFG path intact (needs guidance_scale=base_s > 1 in the pipeline call).
    #     We only reshape the *conditional* half so that after the pipeline's scalar CFG,
    #     the *effective* guidance equals your per-frame ramp g[t] in [g0..g1].
    #     """
    #     import types

    #     orig_forward = pipe.transformer.forward

    #     def framewise_forward(module, *args, **kwargs):
    #         out = orig_forward(*args, **kwargs)

    #         # Accept dict or tensor output (some builds return {"sample": tensor})
    #         if isinstance(out, dict) and "sample" in out:
    #             x = out["sample"]
    #             dict_out = True
    #         else:
    #             x = out
    #             dict_out = False

    #         B2, C, T, H, W = x.shape
    #         if (B2 % 2) != 0:
    #             # Not in CFG mode → leave untouched (e.g., guidance_scale==1.0)
    #             return out

    #         B = B2 // 2
    #         x = x.view(2, B, C, T, H, W)  # [uncond, cond_raw]
    #         x_uncond, x_cond_raw = x[0], x[1]

    #         # Per-frame target guidance ramp g[t]
    #         g = torch.linspace(g0, g1, T, device=x.device, dtype=x.dtype).view(
    #             1, 1, T, 1, 1
    #         )

    #         # Pre-adjust conditional so downstream 'base_s' scalar yields g[t]
    #         scale = g / float(base_s)
    #         x_cond_adj = x_uncond + scale * (x_cond_raw - x_uncond)

    #         x_out = torch.stack([x_uncond, x_cond_adj], dim=0).view(B2, C, T, H, W)
    #         if dict_out:
    #             out["sample"] = x_out
    #             return out
    #         return x_out

    #     pipe.transformer.forward = types.MethodType(framewise_forward, pipe.transformer)
    #     return pipe

    # if input_video_mask is not None:
    #     T = input_video_mask.shape[2]
    #     decay = torch.linspace(
    #         1.0, 0.0, T, device=input_video_mask.device, dtype=input_video_mask.dtype
    #     )
    #     input_video_mask = input_video_mask * decay.view(1, 1, T, 1, 1)
    # # Example ramp: gentle at early frames, strong at the end
    # make_framewise_cfg_wrapper(pipeline, g0=1.0, g1=7.0)
    # BASE_S = 3.0

    attn_mod = pipeline.transformer.blocks[-1].cross_attn

    # # A) BEFORE steering
    # pipeline.transformer.disable_text_spatial_steer()
    # sample = pipeline(
    #     prompt,
    #     num_frames=video_length,
    #     negative_prompt=negative_prompt,
    #     height=sample_size[0],
    #     width=sample_size[1],
    #     generator=generator,
    #     guidance_scale=guidance_scale,
    #     num_inference_steps=num_inference_steps,
    #     boundary=boundary,
    #     video=input_video,
    #     mask_video=input_video_mask,
    #     shift=shift,
    # ).videos
    # save_zebra_heatmaps(attn_mod, out_dir="debug_attn", prefix="before")

    full = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=pipeline.transformer.text_len,
        return_tensors="pt",
    )
    full_ids = full.input_ids[0].tolist()

    # subword IDs for the concept (strip special tokens if any)
    zebra_sub_ids = [
        tid
        for tid in tokenizer("zebra").input_ids
        if tid
        not in (
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            0,
            None,
        )
    ]

    # positions in the full prompt that match any zebra subword
    zebra_pos = [i for i, tid in enumerate(full_ids) if tid in set(zebra_sub_ids)]

    # enable steering with **positions**
    pipe_tr = pipeline.transformer
    pipe_tr.enable_text_spatial_steer(token_ids=zebra_pos, beta=0.7, ema=0.8, topk=0.40)
    # optional extras
    pipe_tr.steering.update({"delta": 3.0, "frac_push": 0.10, "smooth": 0.25})
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
        video=input_video,
        mask_video=input_video_mask,
        shift=shift,
    ).videos
    # save_zebra_heatmaps(attn_mod, out_dir="debug_attn", prefix="after")

if lora_path is not None:
    pipeline = unmerge_lora(
        pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype
    )
    if transformer_2 is not None:
        pipeline = unmerge_lora(
            pipeline,
            lora_high_path,
            lora_high_weight,
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )


def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)


if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist

    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
