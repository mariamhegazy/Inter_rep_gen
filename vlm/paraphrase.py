#!/usr/bin/env python3
# paraphrase_prompts_with_qwen_vl.py
import argparse
import json
import os
from typing import Any, Dict, List

os.environ["HF_HOME"] = "/capstor/scratch/cscs/mhasan/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("SLURM_LOCALID", "0"))

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# === SLURM parallel setup ===
rank = int(os.environ.get("SLURM_PROCID", "0"))
world_size = int(os.environ.get("SLURM_NTASKS", "1"))

# === Model ===
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# The instruction given to Qwen. It receives BOTH the image and the user's prompt.
# We constrain output to ONLY the rewritten text, same meaning, no extra facts.
PARAPHRASE_SYSTEM_PROMPT = (
    "You will be given an image and an accompanying text prompt.\n"
    "Rewrite (paraphrase) the text so it conveys the SAME meaning.\n"
    "Do not add or remove facts, do not mention this is a paraphrase, "
    "and do not include any preface or explanation. Output ONLY the rewritten prompt."
)


def load_items(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    return data


def resolve_path(fname: str, image_root: str = None) -> str:
    if os.path.isabs(fname):
        return fname
    if image_root is not None:
        return os.path.join(image_root, fname)
    return fname


def shard(items: List[Any], rank: int, world_size: int) -> List[Any]:
    n = len(items)
    per = n // world_size
    rem = n % world_size
    start = per * rank + min(rank, rem)
    end = start + per + (1 if rank < rem else 0)
    return items[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON (list of objects with at least {filename, prompt}).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to write the output JSON.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root dir to prepend to filenames if they are relative.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for paraphrase.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.0 = deterministic).",
    )
    args = parser.parse_args()

    # Load data and shard across SLURM ranks
    all_items = load_items(args.input_json)
    local_items = shard(all_items, rank, world_size)
    print(f"🧠 Rank {rank}/{world_size} handling {len(local_items)} items...")

    # Load model + processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    results = []
    for item in tqdm(local_items, desc=f"🖼️ Rank {rank} paraphrasing"):
        if not isinstance(item, dict):
            # Keep structure resilient
            continue

        filename = item.get("file_name")
        raw_prompt = item.get("caption")

        if not filename or raw_prompt is None:
            # Skip malformed entries
            continue

        img_path = resolve_path(filename, args.image_root)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Rank {rank}: failed to open image '{img_path}': {e}")
            # Pass through with a note
            out_entry = dict(item)
            out_entry["paraphrased_prompt"] = None
            out_entry["error"] = f"failed_to_open_image: {str(e)}"
            results.append(out_entry)
            continue

        # Build messages for Qwen (image + instruction + user prompt)
        # Format: system instruction, then user provides image + text prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": PARAPHRASE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"Original prompt:\n{raw_prompt}",
                    },
                ],
            },
        ]

        # Prepare model inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,  # None here since we're using images
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        # Generate paraphrase
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                repetition_penalty=1.0,
            )
            trimmed = output_ids[:, inputs.input_ids.shape[1] :]
            paraphrase = processor.batch_decode(trimmed, skip_special_tokens=True)[
                0
            ].strip()

        # Collect result: keep all original fields and add paraphrased_prompt
        out_entry = dict(item)
        out_entry["paraphrased_prompt"] = paraphrase
        results.append(out_entry)

    # Merge with other ranks: simple approach is one file per rank, or let each rank write full JSON then merge externally.
    # Here we write one file per rank for safety if world_size>1; if single rank, write to args.output_json directly.
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    if world_size == 1:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(results)} items to {args.output_json}")
    else:
        # Write rank-part file; you can merge later.
        base, ext = os.path.splitext(args.output_json)
        rank_path = f"{base}.rank{rank}{ext or '.json'}"
        with open(rank_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Rank {rank} saved {len(results)} items to {rank_path}")


if __name__ == "__main__":
    main()
