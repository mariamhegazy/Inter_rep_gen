#!/usr/bin/env python3
# captions_to_frame_descriptions_llm.py
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Cache + device selection (kept same behavior)
os.environ["HF_HOME"] = "/capstor/scratch/cscs/mhasan/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("SLURM_LOCALID", "0"))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# === SLURM parallel setup ===
rank = int(os.environ.get("SLURM_PROCID", "0"))
world_size = int(os.environ.get("SLURM_NTASKS", "1"))

# === Default text-only LLM (LOCAL PATH) ===
DEFAULT_MODEL_ID = "/capstor/scratch/cscs/mhasan/.cache/hf/Qwen2.5-32B-Instruct"

# Instruction for turning a video-generation prompt into first/middle/last frame captions.
FRAME_DESC_SYSTEM_PROMPT = (
    "You convert a single video-generation prompt into three still-frame captions.\n"
    "\n"
    "STRICT RULES:\n"
    "1) Use ONLY facts explicitly present in the original prompt. Do NOT add, infer, or guess any new details.\n"
    "2) If something (objects, scene, lighting, time of day, weather, camera movement, actions) is not stated, OMIT it. Do not assume defaults.\n"
    "3) Keep wording faithful to the prompt: paraphrase minimally; do not introduce new nouns, adjectives, places, brands, numbers, or styles.\n"
    "4) Maintain temporal continuity using only prompt information: 'first' is the start state, 'middle' a plausible midpoint without new facts, 'last' the end state. If the prompt lacks temporal cues, keep content constant across frames.\n"
    "5) Each caption is 1–2 short sentences; be clear and literal rather than imaginative.\n"
    "\n"
    'Return STRICT JSON (no code fences, no extra text): {"first": str, "middle": str, "last": str}'
)


def load_items(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    return data


def shard(items: List[Any], rank: int, world_size: int) -> List[Any]:
    n = len(items)
    per = max(1, n // world_size) if world_size > 0 else n
    rem = n % world_size if world_size > 0 else 0
    start = per * rank + min(rank, rem)
    end = start + per + (1 if rank < rem else 0)
    return items[start:end]


def maybe_set_pad_token(tokenizer):
    # Some Qwen tokenizers have no pad_token set; tie it to eos to avoid warnings.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_tokenizer_and_model(model_id: str, local_only: bool = True):
    if os.path.isdir(model_id):
        local_only = True  # force local files for local dirs
        print(
            f"[info] Loading model from local directory: {model_id} (local_files_only=True)",
            file=sys.stderr,
        )
    else:
        print(
            f"[info] Loading model by repo id: {model_id} (local_files_only={local_only})",
            file=sys.stderr,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    maybe_set_pad_token(tokenizer)

    # Try with FlashAttention-2 first; if it fails, retry without it.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=local_only,
        )
    except Exception as e:
        print(
            f"[warn] FlashAttention load failed: {e}\n[info] retrying without attn_implementation...",
            file=sys.stderr,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_only,
        )

    return tokenizer, model


def build_messages_for_frames(video_prompt: str):
    return [
        {"role": "system", "content": FRAME_DESC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Video-generation prompt:\n"
                f"{video_prompt}\n\n"
                "Respond with STRICT JSON only."
            ),
        },
    ]


def parse_json_strict(s: str) -> Optional[Dict[str, Any]]:
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Strip markdown fences if present
    stripped = s.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        # remove leading 'json' label if present
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    # Try to extract the first {...} block heuristically
    try:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(stripped[first : last + 1])
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON (list of objects with at least {'caption'}).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to write the output JSON.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Local path or HF model id (default is local Qwen2.5-32B-Instruct).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0.0 = deterministic).",
    )
    parser.add_argument(
        "--no_local_files_only",
        action="store_true",
        help="Allow fetching from the hub if needed (defaults to local-only).",
    )
    args = parser.parse_args()

    # Load and shard work
    all_items = load_items(args.input_json)
    local_items = shard(all_items, rank, world_size)
    print(f"🧠 Rank {rank}/{world_size} handling {len(local_items)} items...")

    # Load tokenizer + LLM (prefer local files)
    tokenizer, model = load_tokenizer_and_model(
        args.model_id, local_only=not args.no_local_files_only
    )

    results = []
    for item in tqdm(local_items, desc=f"🎬 Rank {rank} frame descriptions"):
        if not isinstance(item, dict):
            continue
        raw_prompt = item.get("caption")
        if raw_prompt is None:
            continue

        # Build chat messages for first/middle/last captions
        messages = build_messages_for_frames(raw_prompt)

        # Use the model's chat template
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=(args.temperature if args.temperature > 0 else None),
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
        out_text = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        parsed = parse_json_strict(out_text)
        if parsed is None or not all(k in parsed for k in ("first", "middle", "last")):
            # Fallback to safe placeholders if the model didn't return clean JSON
            parsed = parsed or {}
            print(
                f"[warn] JSON parse failed on rank {rank}; storing raw output.",
                file=sys.stderr,
            )

        out_entry = dict(item)
        out_entry["frame_first"] = parsed.get("first")
        out_entry["frame_middle"] = parsed.get("middle")
        out_entry["frame_last"] = parsed.get("last")
        out_entry["frame_json_raw"] = (
            out_text  # keep raw in case you want to re-parse later
        )
        # keep original caption as-is in the record
        results.append(out_entry)

    # Save (single file if world_size==1; else one file per rank)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    if world_size == 1:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(results)} items to {args.output_json}")
    else:
        base, ext = os.path.splitext(args.output_json)
        rank_path = f"{base}.rank{rank}{ext or '.json'}"
        with open(rank_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Rank {rank} saved {len(results)} items to {rank_path}")


if __name__ == "__main__":
    main()
