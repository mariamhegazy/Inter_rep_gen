#!/usr/bin/env python3
# captions_to_frame_descriptions_llm_both_subjects.py
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

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

# Main instruction: derive first/middle/last while keeping both subjects visible
# and the first frame already beginning both actions.
FRAME_DESC_SYSTEM_PROMPT_BASE = (
    "You convert a video-generation prompt into three still-frame captions.\n"
    "- Use only information present or strongly implied by the prompt.\n"
    "- Be specific about layout, subjects, actions, lighting/time/weather, and camera.\n"
    "- Maintain temporal continuity: FIRST is the start (both subjects already present and beginning their actions), "
    "MIDDLE is a plausible midpoint, LAST is the end state.\n"
    "- You must include BOTH subjects in EVERY frame by their EXACT given words.\n"
    "- Keep each caption 1–2 sentences, natural and vivid.\n"
    "- You may introduce minimal framing (e.g., 'in the background', 'through a window') to place both subjects in one shot, "
    "but do not introduce new subjects or change the actions.\n"
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
    per = n // world_size
    rem = n % world_size
    start = per * rank + min(rank, rem)
    end = start + per + (1 if rank < rem else 0)
    return items[start:end]


def maybe_set_pad_token(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_tokenizer_and_model(model_id: str, local_only: bool = True):
    if os.path.isdir(model_id):
        local_only = True
        print(
            f"[info] Loading model from local directory: {model_id} (local_files_only=True)",
            file=sys.stderr,
        )
    else:
        print(
            f"[info] Loading model by repo id: {model_id} (local_files_only={local_only})",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, local_files_only=local_only
    )
    maybe_set_pad_token(tokenizer)

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


# --- Subject / action helpers ------------------------------------------------

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _clean_q(s: str) -> str:
    s = s.strip()
    if s.endswith("?"):
        s = s[:-1]
    return s.strip()


def _strip_article(s: str) -> str:
    return _ARTICLE_RE.sub("", s).strip()


def extract_subject_and_action(
    phrase_list: Optional[List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristics:
      - subject: shortest string (like 'a dog?') → strip ?, strip article → 'dog'
      - action:  longest string (like 'a dog runs through a field?') → remove article+subject prefix if present
    """
    if not phrase_list or not isinstance(phrase_list, list):
        return None, None

    cleaned = [_clean_q(p) for p in phrase_list if isinstance(p, str) and p.strip()]
    if not cleaned:
        return None, None

    shortest = min(cleaned, key=len)
    longest = max(cleaned, key=len)

    subj = _strip_article(shortest).strip()
    # action: remove leading article and possible subject token(s)
    act = _strip_article(longest)
    # remove subject at start if present
    if subj and act.lower().startswith(subj.lower()):
        act = act[len(subj) :].lstrip(",.:- ").strip()
    # also try removing with leading article+subject (e.g., 'a dog')
    art_subj = _strip_article(shortest)
    if art_subj and act.lower().startswith(art_subj.lower()):
        act = act[len(art_subj) :].lstrip(",.:- ").strip()

    # If action is empty or equals subject, just return None for action
    if not act or act.lower() == subj.lower():
        act = None

    return (subj or None), (act or None)


def subjects_from_item(
    item: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    s0, a0 = extract_subject_and_action(item.get("phrase_0"))
    s1, a1 = extract_subject_and_action(item.get("phrase_1"))
    return s0, a0, s1, a1


# --- Prompt building / validation -------------------------------------------


def build_messages(
    video_prompt: str,
    s0: Optional[str],
    a0: Optional[str],
    s1: Optional[str],
    a1: Optional[str],
) -> List[Dict[str, str]]:
    subjects_line = ""
    actions_line = ""
    if s0 and s1:
        subjects_line = f"Subjects (use these EXACT words in every frame): {s0}; {s1}."
    if a0 or a1:
        parts = []
        if a0 and s0:
            parts.append(f"{s0}: {a0}")
        if a1 and s1:
            parts.append(f"{s1}: {a1}")
        if parts:
            actions_line = "Actions (do not change): " + " | ".join(parts)

    sys_prompt = FRAME_DESC_SYSTEM_PROMPT_BASE
    user_prompt = (
        f"Video-generation prompt:\n{video_prompt}\n\n"
        f"{subjects_line}\n{actions_line}\n"
        "Constraints:\n"
        "- In EVERY frame mention BOTH subjects by those exact words.\n"
        "- In the FIRST frame, BOTH subjects are already present and beginning their actions.\n"
        "- Respond with STRICT JSON only."
    ).strip()

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_strict(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        pass
    stripped = s.strip()
    if stripped.startswith("```"):
        # remove code fence + optional 'json'
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    try:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(stripped[first : last + 1])
    except Exception:
        pass
    return None


def mentions_all_subjects(text: str, subjects: List[str]) -> bool:
    t = text.lower()
    return all((subj or "").lower() in t for subj in subjects if subj)


def validate_frames(parsed: Dict[str, Any], required_subjects: List[str]) -> bool:
    if not parsed or not all(k in parsed for k in ("first", "middle", "last")):
        return False
    return (
        mentions_all_subjects(str(parsed["first"]), required_subjects)
        and mentions_all_subjects(str(parsed["middle"]), required_subjects)
        and mentions_all_subjects(str(parsed["last"]), required_subjects)
    )


def repair_messages(
    prev_out: str, video_prompt: str, s0: Optional[str], s1: Optional[str]
) -> List[Dict[str, str]]:
    subj_line = ""
    if s0 and s1:
        subj_line = f"Your previous output failed to mention both subjects everywhere. You must include BOTH '{s0}' AND '{s1}' explicitly in EVERY frame."
    return [
        {"role": "system", "content": FRAME_DESC_SYSTEM_PROMPT_BASE},
        {
            "role": "user",
            "content": f"{subj_line}\n"
            f"Original video-generation prompt:\n{video_prompt}\n\n"
            "Regenerate STRICT JSON only (no extra text).",
        },
        {
            "role": "assistant",
            "content": prev_out[:4000],
        },  # give a hint of prior structure
    ]


# --- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON (list with at least {'caption'}; optional {'phrase_0','phrase_1'}).",
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="Where to write the output JSON."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Local path or HF model id (default is local Qwen2.5-32B-Instruct).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=220, help="Max new tokens for generation."
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
        help="Allow hub fetching if needed (defaults to local-only).",
    )
    parser.add_argument(
        "--max_regen_attempts",
        type=int,
        default=2,
        help="Max attempts to auto-repair outputs that miss subjects.",
    )
    args = parser.parse_args()

    all_items = load_items(args.input_json)
    local_items = shard(all_items, rank, world_size)
    print(f"🧠 Rank {rank}/{world_size} handling {len(local_items)} items...")

    tokenizer, model = load_tokenizer_and_model(
        args.model_id, local_only=not args.no_local_files_only
    )

    results = []
    for item in tqdm(local_items, desc=f"🎬 Rank {rank} frame descriptions"):
        if not isinstance(item, dict):
            continue
        video_prompt = item.get("caption")
        if not video_prompt:
            continue

        # Extract subjects/actions if available
        s0, a0, s1, a1 = subjects_from_item(item)
        required_subjects = [s for s in [s0, s1] if s]

        # Build initial messages
        messages = build_messages(video_prompt, s0, a0, s1, a1)

        def generate_from_messages(
            msgs: List[Dict[str, str]],
        ) -> Tuple[str, Optional[Dict[str, Any]]]:
            chat_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
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
            out_text = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[
                0
            ].strip()
            parsed = parse_json_strict(out_text)
            return out_text, parsed

        out_text, parsed = generate_from_messages(messages)

        # Validate presence of BOTH subjects in EVERY frame (if we know them)
        valid = True
        if required_subjects:
            valid = validate_frames(parsed or {}, required_subjects)

        # Attempt repair if invalid
        attempts = 0
        while (not valid) and (attempts < args.max_regen_attempts):
            attempts += 1
            repair_msg = repair_messages(out_text, video_prompt, s0, s1)
            out_text, parsed = generate_from_messages(repair_msg)
            valid = validate_frames(parsed or {}, required_subjects)

        # Store outputs
        out_entry = dict(item)
        if parsed and all(k in parsed for k in ("first", "middle", "last")):
            out_entry["frame_first"] = parsed["first"]
            out_entry["frame_middle"] = parsed["middle"]
            out_entry["frame_last"] = parsed["last"]
        else:
            # fallback: keep raw so you can re-parse later
            out_entry["frame_first"] = None
            out_entry["frame_middle"] = None
            out_entry["frame_last"] = None

        out_entry["frame_json_raw"] = out_text
        if required_subjects:
            out_entry["frame_subjects_used"] = required_subjects
            out_entry["frame_validation_ok"] = bool(valid)
            out_entry["frame_attempts"] = 1 + attempts

        results.append(out_entry)

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
