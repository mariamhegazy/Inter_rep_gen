#!/usr/bin/env python3
# augment_prompts_with_qwen_vl.py
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

os.environ["HF_HOME"] = "/capstor/scratch/cscs/mhasan/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("SLURM_LOCALID", "0"))

import re

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

# ---------------------------
# Ontology / policy
# ---------------------------
ALLOWED_CATEGORIES = {
    "architecture",
    "landscape",
    "vehicle",
    "animal",
    "human",
    "cooking",
    "appearance",
    "space",
    "generic",
    "abstract",
}

CAMERA_MOTIONS = [
    "slow pan to the left",
    "slow pan to the right",
    "gentle tilt up",
    "gentle tilt down",
    "slow dolly-in",
    "slow dolly-out",
    "gradual zoom-in",
    "gradual zoom-out",
]

SCENE_TIMES = [
    "dawn",
    "sunrise",
    "noon",
    "afternoon",
    "sunset",
    "twilight",
    "night",
    "midnight",
]

WEATHERS = [
    "fine drizzle falls",
    "raindrops fall steadily",
    "a thunderstorm begins",
    "hail begins to fall",
    "snowflakes fall",
    "dense fog rolls in",
    "it becomes foggy",
    "a hailstorm begins",
]

RELATIONS = {
    "relation_left": "one subject moves to the left of another",
    "relation_right": "one subject moves to the right of another",
    "relation_front": "one subject moves in front of another",
    "relation_back": "one subject moves behind another",
    "relation_next_to": "one subject moves next to another",
}

# Subject-appropriate actions (used to constrain Qwen)
ACTIONS_BY_TYPE = {
    "architecture": [],
    "indoor": [
        "the door opens",
        "the door closes",
        "the lights turn on",
        "the lights turn off",
        "curtains sway slightly",
    ],
    "scenery": ["waves intensify", "mist rises near the waterfall"],
    "landscape": ["waves intensify", "mist rises near the waterfall"],
    "transportation": [
        "accelerates",
        "brakes suddenly",
        "turns left",
        "turns right",
        "overtakes another vehicle",
        "drifts",
    ],
    "vehicle": [
        "accelerates",
        "brakes suddenly",
        "turns left",
        "turns right",
        "overtakes another vehicle",
        "drifts",
    ],
    "animal": [
        "walks",
        "runs",
        "jumps",
        "dives down",
        "swims faster",
        "flaps its wings",
        "looks around",
        "drops what it holds",
    ],
    "single-human": [
        "starts walking",
        "starts running",
        "jumps",
        "waves at the camera",
        "points forward",
        "turns to look at the camera",
        "smiles and nods",
        "claps hands",
        "sits down",
        "stands up",
    ],
    "multiple-human": [
        "start walking",
        "start running",
        "jump",
        "wave at the camera",
        "point forward",
        "turn to look at the camera",
        "smile and nod",
        "clap hands",
        "sit down",
        "stand up",
    ],
    "human": [
        "starts walking",
        "starts running",
        "jumps",
        "waves at the camera",
        "points forward",
        "turns to look at the camera",
        "smiles and nods",
        "claps hands",
        "sits down",
        "stands up",
    ],
    "food": [
        "opens",
        "closes",
        "tilts",
        "pours",
        "whisks",
        "sizzles",
        "steam rises",
        "falls over",
        "starts spinning",
    ],
    "cooking": [
        "opens",
        "closes",
        "tilts",
        "pours",
        "whisks",
        "sizzles",
        "steam rises",
        "falls over",
        "starts spinning",
    ],
    "plant": ["sways gently", "petals open"],
    "appearance": [],  # handled via color changes
    "space": [
        "slowly rotates",
        "the camera slowly rolls",
        "the terminator line on Earth shifts",
    ],
}

# Mapping of augment_type to short templates Qwen can use
AUGMENT_DIMENSIONS = {
    "camera_motion": "The camera performs a {camera_motion}.",
    "scene_time": "It gradually becomes {scene_time}.",
    "scene_weather": "Next, {weather}.",
    "appearance_color_change": "The main subject becomes {color} by the end.",
    "relation_left": "Then {relation_left}.",
    "relation_right": "Then {relation_right}.",
    "relation_front": "Then {relation_front}.",
    "relation_back": "Then {relation_back}.",
    "relation_next_to": "Then {relation_next_to}.",
    "action": "Then {action}.",  # action is category-aware
}

COLORS = [
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "orange",
    "gold",
    "silver",
    "black",
    "white",
    "purple",
    "pink",
    "gray",
    "brown",
]

# System prompt for Qwen: precise JSON-only output with 3 variants
SYSTEM_PROMPT = (
    "You will receive an image plus metadata: base caption, type, category, and an optional augment_type.\n"
    "Generate EXACTLY THREE modified prompts that are short, grammatical continuations of the base caption, suitable for text-to-video.\n"
    "Each modification must be plausible and consistent with the scene and with the subject's capabilities.\n"
    "Follow these rules:\n"
    "1) If augment_type is provided, respect it (e.g., camera_motion, scene_time, scene_weather, appearance_color_change, relation_*, action).\n"
    "2) If augment_type is 'action', choose an action appropriate to the given type/category (e.g., vehicles accelerate/turn; humans wave/point; animals swim/fly/jump; buildings do not act).\n"
    "3) Weather in space is not rain/snow/hail; instead use camera roll, rotation, or Earth terminator shifts.\n"
    "4) Relations (left/right/in front/behind/next to) require at least two entities; if the base caption shows only one, introduce a second entity briefly (e.g., 'another boat enters frame'), minimally and plausibly.\n"
    "5) Appearance color changes should not rename or replace objects; just describe a color shift.\n"
    "6) Prefer explicit agent names to avoid ambiguity ('the camera pans right' vs 'turns right').\n"
    "7) Do NOT contradict the base caption.\n"
    "Output ONLY valid JSON with this schema:\n"
    "{\n"
    '  "variants": [\n'
    '    {"caption_aug": "<string>", "augment_type": "<string>", "notes": "<very short why/constraints>"} ,\n'
    '    {"caption_aug": "...", "augment_type": "...", "notes": "..."},\n'
    '    {"caption_aug": "...", "augment_type": "...", "notes": "..."}\n'
    "  ]\n"
    "}\n"
)


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    return data


def resolve_path(fname: str, image_root: str = None) -> str:
    if os.path.isabs(fname):
        return fname
    return os.path.join(image_root, fname) if image_root else fname


def shard(items: List[Any], rank: int, world_size: int) -> List[Any]:
    n = len(items)
    per = n // world_size
    rem = n % world_size
    start = per * rank + min(rank, rem)
    end = start + per + (1 if rank < rem else 0)
    return items[start:end]


def pick_defaults(augment_type: str) -> Dict[str, str]:
    # Provide defaults Qwen can choose from if it wants placeholders
    out = {}
    if augment_type == "camera_motion":
        out["camera_motion"] = CAMERA_MOTIONS[0]
    elif augment_type == "scene_time":
        out["scene_time"] = SCENE_TIMES[0]
    elif augment_type == "scene_weather":
        out["weather"] = WEATHERS[0]
    elif augment_type == "appearance_color_change":
        out["color"] = COLORS[0]
    return out


def build_user_prompt(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    caption = item.get("caption") or item.get("original_caption") or ""
    typ = (item.get("type") or "").lower()
    category = (item.get("category") or "generic").lower()
    augment_type = item.get("augment_type")  # may be None

    # Compose a short metadata block Qwen can read
    meta_lines = [
        f"Base caption: {caption}",
        f"type: {typ}",
        f"category: {category}",
        f"augment_type: {augment_type or 'auto'}",
    ]

    # Provide small, explicit option lists to anchor generations
    # (Qwen is better behaved when we show allowed verbs per category)
    action_list = ACTIONS_BY_TYPE.get(typ, []) or ACTIONS_BY_TYPE.get(category, [])
    if augment_type == "action" and not action_list:
        # Fall back to camera/time to avoid impossible actions
        meta_lines.append("hint: prefer camera/time; subject likely static.")

    if action_list:
        meta_lines.append(f"allowed_actions: {', '.join(action_list)}")

    if augment_type in RELATIONS:
        meta_lines.append(
            "relations_rule: ensure two entities are present; add a minimal second entity if needed."
        )

    if category == "space" or ("space" in caption.lower()):
        meta_lines.append(
            "space_rule: forbid rain/snow/hail; prefer camera roll, rotation, terminator shifts."
        )

    # Provide small vocab for colors/time/weather/camera
    meta_lines.append(f"camera_options: {', '.join(CAMERA_MOTIONS[:6])}")
    meta_lines.append(f"time_options: {', '.join(SCENE_TIMES)}")
    meta_lines.append(
        f"weather_options: foggy, drizzle, rain, thunderstorm, hail, snow"
    )
    meta_lines.append(f"color_options: {', '.join(COLORS)}")

    # Minimal exemplar to coax terse outputs
    meta_lines.append(
        "style: keep additions short, append as a second sentence where possible."
    )

    # Final text
    meta_text = "\n".join(meta_lines)

    return [{"type": "text", "text": meta_text}]


def parse_qwen_json(s: str) -> Dict[str, Any]:
    # Qwen usually outputs clean JSON with our schema; guard for trailing text.
    m = re.search(r"\{[\s\S]*\}$", s.strip())
    if not m:
        raise ValueError("Model did not return JSON.")
    obj = json.loads(m.group(0))
    if "variants" not in obj or not isinstance(obj["variants"], list):
        raise ValueError("JSON missing 'variants' list.")
    return obj


def basic_validate(
    item: Dict[str, Any], variants: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[str]]:
    errors = []
    caption = item.get("caption", "")
    typ = (item.get("type") or "").lower()
    category = (item.get("category") or "generic").lower()
    aug_req = item.get("augment_type")

    def has_two_entities(text: str) -> bool:
        # very rough heuristic
        tokens = [
            " and ",
            " another ",
            " two ",
            " three ",
            " second ",
            " another",
            " others",
        ]
        return any(t in text.lower() for t in tokens)

    cleaned = []
    for v in variants:
        cap = v.get("caption_aug", "").strip()
        a_type = v.get("augment_type", "").strip() or (aug_req or "auto")

        # Force explicit camera subject to avoid ambiguity
        if "turns left" in cap or "turns right" in cap:
            if "camera" not in cap and typ not in (
                "vehicle",
                "transportation",
                "human",
                "single-human",
                "multiple-human",
            ):
                cap = cap.replace("Then turns", "The camera turns")

        # Relations need 2 entities
        if a_type.startswith("relation"):
            if not has_two_entities(caption) and not has_two_entities(cap):
                cap = cap + " Another object enters the frame, enabling the relation."

        # Space weather sanity
        if category == "space" or "space" in caption.lower():
            if any(w in cap.lower() for w in ["rain", "snow", "hail", "drizzle"]):
                cap = re.sub(
                    r"(rain|snow|hail|drizzle)[^\.\!]*",
                    "the camera slowly rolls",
                    cap,
                    flags=re.I,
                )

        cleaned.append(
            {"caption_aug": cap, "augment_type": a_type, "notes": v.get("notes", "")}
        )

    return cleaned, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    all_items = load_items(args.input_json)
    local_items = shard(all_items, rank, world_size)
    print(f"🧠 Rank {rank}/{world_size} handling {len(local_items)} items...")

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    results = []
    for item in tqdm(local_items, desc=f"🎞️ Rank {rank} augmenting"):
        if not isinstance(item, dict):
            continue

        fname = item.get("file_name") or item.get("filename")
        if not fname or item.get("caption") is None:
            continue

        img_path = resolve_path(fname, args.image_root)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            out_entry = dict(item)
            out_entry["augmented"] = None
            out_entry["error"] = f"failed_to_open_image: {str(e)}"
            results.append(out_entry)
            continue

        # Messages: system + user (image + metadata text)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    *build_user_prompt(item),
                ],
            },
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                repetition_penalty=1.05,
            )
            trimmed = output[:, inputs.input_ids.shape[1] :]
            raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[
                0
            ].strip()

        # Parse + validate
        out_entry = dict(item)
        try:
            parsed = parse_qwen_json(raw_text)
            cleaned, errs = basic_validate(item, parsed.get("variants", []))
            out_entry["augmented"] = (
                cleaned  # list of 3 dicts {caption_aug, augment_type, notes}
            )
            if errs:
                out_entry["validation_warnings"] = errs
        except Exception as e:
            out_entry["augmented"] = None
            out_entry["error"] = f"bad_json_from_model: {str(e)}"
            out_entry["raw_model_text"] = raw_text

        results.append(out_entry)

    # Save
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
