import gc
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

from types import MethodType

import torch
from einops import rearrange

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

# ----------------------------
# Config (unchanged)
# ----------------------------
GPU_memory_mode = "sequential_cpu_offload"
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False

enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

cfg_skip_ratio = 0

enable_riflex = False
riflex_k = 6

config_path = "config/wan2.2/wan_civitai_5b.yaml"
model_name = "models/Wan2.2-TI2V-5B"

sampler_name = "Flow_Unipc"
shift = 5

transformer_path = None
transformer_high_path = None
vae_path = None
lora_path = None
lora_high_path = None

sample_size = [704, 1280]
video_length = 121
fps = 24

weight_dtype = torch.bfloat16
validation_image_start = "/capstor/store/cscs/swissai/a144/mariam/vbench2_beta_i2v/data/crop/16-9/an elephant walking through a forest.jpg"

prompt = "a zebra walking through a forest"
negative_prompt = (
    "overexposed, blurry, low quality, deformed hands, ugly, artifacts, static scene"
)
guidance_scale = 6.0
seed = 43
num_inference_steps = 50
lora_weight = 0.55
lora_high_weight = 0.55
save_path = "samples/wan-videos-ti2v"

# ----------------------------
# Build models (unchanged)
# ----------------------------
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
        from safetensors.torch import load_file

        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None and transformer_high_path is not None:
    print(f"From checkpoint: {transformer_high_path}")
    if transformer_high_path.endswith("safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(transformer_high_path)
    else:
        state_dict = torch.load(transformer_high_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer_2.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

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
        from safetensors.torch import load_file

        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

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
        ["modulation"],
        device=device,
    )
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(
            transformer_2,
            ["modulation"],
            device=device,
        )
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=["modulation"],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(
            transformer_2,
            exclude_module_name=["modulation"],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=["modulation"],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(
            transformer_2,
            exclude_module_name=["modulation"],
            device=device,
        )
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

# Only enable cfg_skip when actually used (unchanged)
if cfg_skip_ratio is not None and cfg_skip_ratio > 0:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        transformer_2.share_cfg_skip(transformer=pipeline.transformer)

# ----------------------------
# Helper: replicate get_image_to_video_latent usage (unchanged)
# ----------------------------
with torch.no_grad():
    # make length compatible with VAE temporal compression
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

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# ----------------------------
# Monkey-patched sampler: UNCOND mask=255, COND mask=original
# ----------------------------
from diffusers.utils.torch_utils import randn_tensor

from videox_fun.pipeline.pipeline_wan2_2_ti2v import retrieve_timesteps


def run_with_uncond_all255(
    self: Wan2_2TI2VPipeline,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    generator,
    boundary,
    video,
    mask_video,
    shift,
):
    device = self._execution_device
    weight_dtype = self.text_encoder.dtype
    do_cfg = guidance_scale > 1.0

    # Encode text (same as pipeline)
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=device,
    )
    in_prompt_embeds = (
        negative_prompt_embeds + prompt_embeds if do_cfg else prompt_embeds
    )

    # Timesteps (same)
    if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, mu=1
        )
    elif isinstance(self.scheduler, (FlowUniPCMultistepScheduler,)):
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps
    elif isinstance(self.scheduler, (FlowDPMSolverMultistepScheduler,)):
        from videox_fun.utils.fm_solvers import get_sampling_sigmas

        sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
        timesteps, _ = retrieve_timesteps(
            self.scheduler, device=device, sigmas=sampling_sigmas
        )
    else:
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None
        )
    self._num_timesteps = len(timesteps)

    # Prepare init_video & mask latents exactly as in pipeline
    if video is not None:
        video_length_local = video.shape[2]
        init_video = self.image_processor.preprocess(
            rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width
        )
        init_video = init_video.to(dtype=torch.float32)
        init_video = rearrange(
            init_video, "(b f) c h w -> b c f h w", f=video_length_local
        )
    else:
        init_video = None

    # Sample base latents
    latent_channels = self.vae.config.latent_channels
    latents = randn_tensor(
        (
            1,  # batch
            latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        ),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    if hasattr(self.scheduler, "init_noise_sigma"):
        latents = latents * self.scheduler.init_noise_sigma

    # Prepare masked_video_latents and latent-space mask (same code path)
    masked_video_latents = None
    mask = None
    seq_len = None
    if init_video is not None and not (mask_video == 255).all():
        bs, _, video_length_src, h_src, w_src = video.size()
        mask_condition = self.mask_processor.preprocess(
            rearrange(mask_video, "b c f h w -> (b f) c h w"), height=h_src, width=w_src
        )
        mask_condition = mask_condition.to(dtype=torch.float32)
        mask_condition = rearrange(
            mask_condition, "(b f) c h w -> b c f h w", f=video_length_src
        )

        masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)

        # encode masked video -> latents
        bs_encode = 1
        new_mask_pixel_values = []
        for i in range(0, masked_video.shape[0], bs_encode):
            mv_bs = masked_video[i : i + bs_encode].to(
                device=device, dtype=self.vae.dtype
            )
            mv_lat = self.vae.encode(mv_bs)[0].mode()
            new_mask_pixel_values.append(mv_lat)
        masked_video_latents = torch.cat(new_mask_pixel_values, dim=0)

        # Build latent-space binary mask in shape of VAEs temporal compression
        mask_condition = torch.concat(
            [
                torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
                mask_condition[:, :, 1:],
            ],
            dim=2,
        )
        mask_condition = mask_condition.view(
            bs, mask_condition.shape[2] // 4, 4, h_src, w_src
        )
        mask_condition = mask_condition.transpose(1, 2)
        # Downsample to latent size and binarize to {0,1}
        target_sz = latents.size()[-3:]
        mask = torch.nn.functional.interpolate(
            mask_condition[:, :1], size=target_sz, mode="trilinear", align_corners=True
        ).to(device, weight_dtype)
        # mask: 1 = generate, 0 = keep reference (this matches original pipeline math)

    # Compute seq_len for rotary/patch math (same as pipeline)
    target_shape = (
        self.vae.latent_channels,
        (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
        width // self.vae.spatial_compression_ratio,
        height // self.vae.spatial_compression_ratio,
    )
    seq_len = int(
        torch.ceil(
            torch.tensor(
                (target_shape[2] * target_shape[3])
                / (
                    self.transformer.config.patch_size[1]
                    * self.transformer.config.patch_size[2]
                )
                * target_shape[1]
            )
        ).item()
    )

    # IMPORTANT: do NOT do the pipeline’s pre-step blend here for UNCOND.
    # We’ll inject reference ONLY in the COND half before the forward pass.

    # Denoising loop
    self.transformer.num_inference_steps = num_inference_steps
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            self.transformer.current_steps = i
            # Scale model input
            latent_model_input_base = latents
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input_base = self.scheduler.scale_model_input(
                    latent_model_input_base, t
                )

            # Build per-branch inputs:
            #  - UNCOND: behave as if mask == 1 everywhere  -> NO ref re-injection
            #  - COND:   original behavior -> re-inject (1 - mask) * masked_video_latents
            if do_cfg:
                # start from the same base tensor
                uncond_in = latent_model_input_base
                cond_in = latent_model_input_base
                if (
                    init_video is not None
                    and mask is not None
                    and masked_video_latents is not None
                ):
                    cond_in = (
                        1 - mask
                    ) * masked_video_latents + mask * cond_in  # only cond gets ref
                latent_model_input = torch.cat([uncond_in, cond_in], dim=0)
            else:
                # No CFG: just behave like cond
                if (
                    init_video is not None
                    and mask is not None
                    and masked_video_latents is not None
                ):
                    latent_model_input_base = (
                        1 - mask
                    ) * masked_video_latents + mask * latent_model_input_base
                latent_model_input = latent_model_input_base

            # Build per-sample timesteps (same as pipeline)
            if init_video is not None:
                # mask is latent-space (B,1,T',H',W') in {0,1}; seq_len is the transformer token length
                # ---- cond path: original masked temp_ts ----
                mask_down = mask[0][0][
                    :, ::2, ::2
                ]  # (T', H', W') subsampled to match patching
                temp_ts_cond = (mask_down * t).flatten()
                if temp_ts_cond.numel() < seq_len:
                    temp_ts_cond = torch.cat(
                        [
                            temp_ts_cond,
                            temp_ts_cond.new_full((seq_len - temp_ts_cond.numel(),), t),
                        ]
                    )
                temp_ts_cond = temp_ts_cond.unsqueeze(0)  # (1, seq_len)

                if do_cfg:
                    # ---- uncond path: pretend mask==1 everywhere -> pure t everywhere ----
                    temp_ts_uncond = torch.full_like(temp_ts_cond, t)
                    timestep = torch.cat(
                        [temp_ts_uncond, temp_ts_cond], dim=0
                    )  # (2, seq_len)
                else:
                    timestep = temp_ts_cond  # (1, seq_len)
            else:
                # No I2V / no reference → standard scalar t
                timestep = t.expand(latent_model_input.shape[0] if do_cfg else 1)

            # Choose transformer (same boundary logic)
            if self.transformer_2 is not None:
                if t >= boundary * self.scheduler.config.num_train_timesteps:
                    local_transformer = self.transformer_2
                else:
                    local_transformer = self.transformer
            else:
                local_transformer = self.transformer

            # Forward
            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(
                device=device
            ):
                noise_pred = local_transformer(
                    x=latent_model_input,
                    context=in_prompt_embeds,
                    t=timestep,
                    seq_len=seq_len,
                )

            # CFG combine (same)
            if do_cfg:
                if self.transformer_2 is not None and (
                    isinstance(self.guidance_scale, (list, tuple))
                ):
                    sample_guide_scale = (
                        self.guidance_scale[1]
                        if t
                        >= self.transformer_2.config.boundary
                        * self.scheduler.config.num_train_timesteps
                        else self.guidance_scale[0]
                    )
                else:
                    sample_guide_scale = guidance_scale
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # CRITICAL: DO NOT global re-inject ref after step.
            # (In vanilla pipeline this line would run:)
            #   if init_video is not None: latents = (1 - mask) * masked_video_latents + mask * latents
            # We skip it so UNCOND stays "mask=255" and COND ref is already injected pre-forward.

            if i == len(timesteps) - 1:
                print(
                    f"[step {i+1}/{len(timesteps)}] UNCOND mask = 255, COND mask = original"
                )
            progress_bar.update()

    # Decode (same)
    frames = self.vae.decode(latents.to(self.vae.dtype)).sample
    frames = (frames / 2 + 0.5).clamp(0, 1)

    # Ask the video processor for a torch tensor
    video_pt = self.video_processor.postprocess_video(video=frames, output_type="pt")
    return video_pt


# Bind it
def _call_once(
    prompt,
    negative_prompt,
    height,
    width,
    generator,
    guidance_scale,
    num_inference_steps,
    boundary,
    video,
    mask_video,
    shift,
):
    return run_with_uncond_all255(
        pipeline,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames=video_length,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        boundary=boundary,
        video=video,
        mask_video=mask_video,
        shift=shift,
    )


# ----------------------------
# TeaCache / cfg-skip (unchanged setup)
# ----------------------------
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

# ----------------------------
# Run
# ----------------------------
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
    sample = _call_once(
        prompt=prompt,
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
    )

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


# ----------------------------
# Save (unchanged)
# ----------------------------
def _to_bcthw(x: torch.Tensor) -> torch.Tensor:
    """Normalize video tensor to B,C,T,H,W."""
    assert x.ndim == 5, f"Expected 5D video tensor, got {tuple(x.shape)}"
    B, A, B2, C2, D2 = x.shape  # just to avoid name clash in error msgs
    # Common cases:
    # - B,C,T,H,W  (channels in dim=1)
    # - B,T,C,H,W  (channels in dim=2)
    if x.shape[1] in (1, 3):  # looks like B,C,T,H,W already
        return x
    if x.shape[2] in (1, 3):  # B,T,C,H,W -> B,C,T,H,W
        return x.permute(0, 2, 1, 3, 4).contiguous()
    # Fallback: try to guess based on last-3 dims being (H,W) and a small C
    raise ValueError(
        f"Unexpected video shape {tuple(x.shape)}; can’t infer channels/time dims."
    )


def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # make sure 'sample' is CPU float32 in [0,1] and B,C,T,H,W
    s = sample.detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
    try:
        s = _to_bcthw(s)
    except Exception as e:
        print("Shape normalization failed:", e)
        print("sample shape was:", tuple(sample.shape))
        raise

    print("Saving video with shape (B,C,T,H,W):", tuple(s.shape))

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    if s.shape[2] == 1:  # T == 1 -> single frame
        video_path = os.path.join(save_path, prefix + ".png")
        frame = s[0, :, 0]  # (C,H,W)
        frame = frame.transpose(0, 1).transpose(1, 2)  # -> (H,W,C)
        img = (frame * 255).numpy().astype(np.uint8)
        Image.fromarray(img).save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        # save_videos_grid expects B,C,T,H,W float32 on CPU in [0,1]
        save_videos_grid(s, video_path, fps=fps)

    print("Saved to:", video_path)


sample = sample.detach().cpu()

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist

    if dist.get_rank() == 0:
        save_results()
    else:
        save_results()
else:
    save_results()
