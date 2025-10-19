#!/usr/bin/env python3
# build_aug_and_contradictions.py
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------
# Lexicons
# ---------------------------
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
    "brown",
    "gray",
    "gold",
    "silver",
    "cyan",
    "magenta",
]
# For contradiction color changes we will AVOID picking 'black' explicitly.
COLORS_NO_BLACK = [c for c in COLORS if c.lower() != "black"]

WEATHERS = ["rain", "snow", "fog", "hail", "thunderstorm", "drizzle"]
WEATHER_PHRASES = {
    "rain": ["it starts raining", "raindrops fall steadily"],
    "snow": ["it starts snowing", "snowflakes fall"],
    "fog": ["dense fog rolls in", "it becomes foggy"],
    "hail": ["hail begins to fall", "a hailstorm begins"],
    "thunderstorm": ["a thunderstorm begins", "lightning flashes and thunder rumbles"],
    "drizzle": ["a light drizzle begins", "fine drizzle falls"],
}

# Basic animal list (singular)
ANIMALS = [
    "dog",
    "cat",
    "bird",
    "eagle",
    "bear",
    "tiger",
    "lion",
    "cow",
    "horse",
    "monkey",
    "shark",
    "penguin",
    "turtle",
    "giraffe",
    "zebra",
    "camel",
    "panda",
    "fish",
    "jellyfish",
    "seal",
    "otter",
    "sheep",
    "goat",
    "chicken",
    "rooster",
    "duck",
    "swan",
    "bee",
    "crab",
    "rhino",
    "elephant",
    "dolphin",
    "whale",
    "fox",
    "wolf",
]

# Simple, common “new objects”
NEW_OBJECTS = [
    "umbrella",
    "kite",
    "ball",
    "drone",
    "balloon",
    "bicycle",
    "lantern",
    "flag",
    "skateboard",
    "surfboard",
    "backpack",
    "guitar",
    "camera",
    "parachute",
    "suitcase",
]

HUMAN_ACTIONS = [
    "starts walking",
    "starts running",
    "waves at the camera",
    "smiles and nods",
    "jumps",
    "turns to look at the camera",
    "claps hands",
    "sits down",
    "stands up",
    "points forward",
]
ANIMAL_ACTIONS = [
    "starts running",
    "looks around",
    "jumps",
    "starts eating",
    "flaps its wings",
    "swims faster",
    "dives down",
    "climbs up",
]
VEHICLE_ACTIONS = [
    "starts moving",
    "accelerates",
    "turns left",
    "turns right",
    "drifts",
    "brakes suddenly",
    "overtakes another vehicle",
]
OBJECT_ACTIONS = [
    "falls over",
    "starts spinning",
    "slides to the side",
    "opens",
    "closes",
    "tilts gently",
]
CAM_ACTS = [
    "slow pan to the right",
    "slow pan to the left",
    "slow dolly-in",
    "slow dolly-out",
    "gentle tilt up",
    "gentle tilt down",
]
RELATIONS = [
    ("moves to the left of", "relation_left"),
    ("moves to the right of", "relation_right"),
    ("moves in front of", "relation_front"),
    ("moves behind", "relation_back"),
    ("moves next to", "relation_next_to"),
]

NUMBER_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]
NUMBER_TO_INT = {w: i + 1 for i, w in enumerate(NUMBER_WORDS)}
INT_TO_WORD = {v: k for k, v in NUMBER_TO_INT.items()}

# ---------------------------
# Regex helpers
# ---------------------------
WORD = r"[A-Za-z]+(?:'[A-Za-z]+)?"
COLOR_RE = re.compile(r"\b(" + "|".join(map(re.escape, COLORS)) + r")\b", re.IGNORECASE)
WEATHER_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, WEATHERS)) + r")\b", re.IGNORECASE
)
ANIMAL_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, ANIMALS)) + r")s?\b", re.IGNORECASE
)
COUNT_RE = re.compile(
    r"\b((?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)|\d+)\s+("
    + WORD
    + r")s?\b",
    re.IGNORECASE,
)


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())[:200] or "item"


def has_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def categorize(caption: str) -> str:
    t = caption.lower()
    if has_any(
        t, ["abstract", "smoke", "bubbles", "neon", "paint", "liquid", "texture"]
    ):
        return "abstract"
    if has_any(
        t,
        [
            "cook",
            "fry",
            "boil",
            "kitchen",
            "pan",
            "wok",
            "grill",
            "pot",
            "dumpling",
            "noodle",
            "egg",
        ],
    ):
        return "cooking"
    if has_any(
        t,
        [
            "man",
            "woman",
            "person",
            "people",
            "girl",
            "boy",
            "child",
            "elderly",
            "older",
        ],
    ):
        return "human"
    if ANIMAL_RE.search(caption):
        return "animal"
    if has_any(
        t,
        [
            "car",
            "bus",
            "train",
            "truck",
            "motorcycle",
            "bike",
            "bicycle",
            "boat",
            "ship",
            "airplane",
            "tram",
        ],
    ):
        return "vehicle"
    if has_any(
        t,
        [
            "mountain",
            "forest",
            "waterfall",
            "beach",
            "ocean",
            "lake",
            "river",
            "desert",
            "storm",
            "snow",
            "fog",
        ],
    ):
        return "landscape"
    if has_any(
        t,
        [
            "bridge",
            "castle",
            "church",
            "pagoda",
            "temple",
            "building",
            "city",
            "street",
            "skyline",
            "gallery",
        ],
    ):
        return "architecture"
    if has_any(t, ["space", "galaxy", "astronaut", "space station", "shuttle"]):
        return "space"
    if COLOR_RE.search(caption):
        return "appearance"
    return "generic"


def pick_not_in_caption(caption: str, pool: List[str]) -> str:
    used = set([m.group(0).lower() for m in re.finditer(r"\b\w+\b", caption.lower())])
    candidates = [c for c in pool if c.lower() not in used]
    return random.choice(candidates) if candidates else random.choice(pool)


def ensure_sentence_end(s: str) -> str:
    return s if s.strip().endswith((".", "!", "?")) else s.strip() + "."


def inline_append(sentence: str, phrase: str) -> str:
    """
    Attach a short clause to the SAME sentence (no new sentences).
    """
    s = sentence.rstrip()
    if not s:
        return phrase
    if s[-1] in ".!?":
        return s[:-1] + f", {phrase}{s[-1]}"
    return s + f", {phrase}"


def pluralize(noun: str, n: int) -> str:
    if n == 1:
        return noun
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"
    return noun + "s"


def word_number(n: int) -> str:
    return INT_TO_WORD.get(n, str(n))


# ---------------------------
# AUGMENTATIONS (adds info; can use extra sentence)
# ---------------------------
def build_action_aug(caption: str, cat: str) -> Tuple[str, str]:
    if cat == "human":
        act = random.choice(HUMAN_ACTIONS)
    elif cat == "animal":
        act = random.choice(ANIMAL_ACTIONS)
    elif cat == "vehicle":
        act = random.choice(VEHICLE_ACTIONS)
    else:
        act = random.choice(OBJECT_ACTIONS)
    aug = ensure_sentence_end(caption) + f" Then {act}."
    return aug, "action"


def build_attribute_aug(caption: str) -> Tuple[str, str]:
    color = pick_not_in_caption(caption, COLORS)
    aug = (
        ensure_sentence_end(caption) + f" The main subject becomes {color} by the end."
    )
    return aug, "appearance_color_change"


def build_scene_aug(caption: str) -> Tuple[str, str]:
    choice = random.choice(["weather", "time", "camera"])
    if choice == "weather":
        w = pick_not_in_caption(caption, WEATHERS)
        phr = random.choice(WEATHER_PHRASES[w])
        aug = ensure_sentence_end(caption) + f" Next, {phr}."
        return aug, "scene_weather"
    elif choice == "time":
        t = pick_not_in_caption(
            caption, ["night", "dawn", "sunset", "twilight", "noon", "midnight"]
        )
        aug = ensure_sentence_end(caption) + f" It gradually becomes {t}."
        return aug, "scene_time"
    else:
        cam = random.choice(CAM_ACTS)
        aug = ensure_sentence_end(caption) + f" The camera performs a {cam}."
        return aug, "camera_motion"


def build_relation_aug(caption: str) -> Tuple[str, str]:
    phrase, tag = random.choice(RELATIONS)
    aug = ensure_sentence_end(caption) + f" Then one subject {phrase} another."
    return aug, tag


def build_best_aug(caption: str, cat: str) -> Tuple[str, str]:
    if cat in ["human", "animal", "vehicle", "cooking"]:
        return build_action_aug(caption, cat)
    if cat in ["landscape", "architecture", "space"]:
        return build_scene_aug(caption)
    if cat in ["appearance", "abstract"]:
        return build_attribute_aug(caption)
    return random.choice(
        [
            lambda: build_action_aug(caption, cat),
            lambda: build_attribute_aug(caption),
            lambda: build_scene_aug(caption),
            lambda: build_relation_aug(caption),
        ]
    )()


# ---------------------------
# CONTRADICTIONS (DIRECT in-sentence edits only)
# ---------------------------
def change_existing_color_inline(caption: str) -> Tuple[str, str]:
    """Replace the first color word with a different (non-black) color."""
    m = COLOR_RE.search(caption)
    if m:
        orig = m.group(0)
        # avoid selecting 'black' to sidestep 'subject is black' phrasing/edge-cases
        candidates = [c for c in COLORS_NO_BLACK if c.lower() != orig.lower()]
        new_color = pick_not_in_caption(caption, candidates) if candidates else orig
        edited = caption[: m.start()] + new_color + caption[m.end() :]
        return edited, "color_change"
    else:
        # No color present: append a tiny inline clause (still one sentence)
        new_color = pick_not_in_caption(caption, COLORS_NO_BLACK)
        edited = inline_append(caption, f"featuring {new_color} tones")
        return edited, "color_injection"


def change_weather_inline(caption: str) -> Tuple[str, str]:
    """Replace/add weather inline within the same sentence."""
    if WEATHER_RE.search(caption):
        # replace the first weather token with a different one
        m = WEATHER_RE.search(caption)
        w_old = m.group(0).lower()
        w_new_pool = [w for w in WEATHERS if w != w_old]
        w_new = pick_not_in_caption(caption, w_new_pool) if w_new_pool else w_old
        edited = caption[: m.start()] + w_new + caption[m.end() :]
        return edited, "weather_change"
    else:
        # add short inline weather phrase
        w_new = pick_not_in_caption(caption, WEATHERS)
        edited = inline_append(caption, f"under {w_new}")
        return edited, "weather_injection"


def swap_animal_type_inline(caption: str) -> Tuple[str, str]:
    """Replace animal noun(s) with a different animal (plurality preserved crudely)."""
    m = ANIMAL_RE.search(caption)
    if not m:
        # No animal — fallback to an inline object phrase instead of a new sentence
        obj = pick_not_in_caption(caption, NEW_OBJECTS)
        art = "an" if obj[0].lower() in "aeiou" else "a"
        edited = inline_append(caption, f"with {art} {obj}")
        return edited, "object_insertion"
    found = m.group(0)  # may be plural
    base = re.sub(r"s\b", "", found, flags=re.IGNORECASE)
    candidates = [a for a in ANIMALS if a.lower() != base.lower()]
    new = random.choice(candidates) if candidates else base
    plural = found.lower().endswith("s")
    new_form = new + ("s" if plural else "")
    edited = caption[: m.start()] + new_form + caption[m.end() :]
    return edited, "animal_type_swap"


def change_count_inline(caption: str) -> Tuple[str, str]:
    """
    Replace first 'count + noun' with a different number; adjust plural inline.
    If none present, add a minimal inline count phrase.
    """
    m = COUNT_RE.search(caption)
    if not m:
        noun = pick_not_in_caption(
            caption,
            ["person", "car", "bird", "flower", "tree", "balloon", "horse", "dog"],
        )
        n_new = random.choice([2, 3, 4])
        phrase = f"showing {word_number(n_new)} {pluralize(noun, n_new)}"
        edited = inline_append(caption, phrase)
        return edited, "count_injection"
    num_txt = m.group(1)
    noun = m.group(2)
    n = int(num_txt) if num_txt.isdigit() else NUMBER_TO_INT.get(num_txt.lower(), 1)
    choices = [k for k in [1, 2, 3, 4, 5] if k != n]
    n_new = random.choice(choices) if choices else (n + 1)
    num_new = word_number(n_new)
    noun_new = pluralize(noun, n_new)
    edited = caption[: m.start()] + f"{num_new} {noun_new}" + caption[m.end() :]
    return edited, "count_change"


def insert_object_inline(caption: str) -> Tuple[str, str]:
    """Inline object insertion (no new sentence)."""
    obj = pick_not_in_caption(caption, NEW_OBJECTS)
    art = "an" if obj[0].lower() in "aeiou" else "a"
    edited = inline_append(caption, f"with {art} {obj}")
    return edited, "object_insertion"


def build_contradiction(caption: str, cat: str) -> Tuple[str, str]:
    """
    Direct, in-sentence edits only.
    - Architecture: stick to weather edits (no color changes).
    - Others: favor animal/type/count; color allowed (avoid 'black').
    """
    if cat == "architecture":
        # ONLY weather edits for architecture
        return change_weather_inline(caption)

    if cat == "animal":
        r = random.random()
        if r < 0.6:
            return swap_animal_type_inline(caption)
        if r < 0.8:
            return change_count_inline(caption)
        if r < 0.9:
            return change_existing_color_inline(caption)
        return change_weather_inline(caption)

    if cat in [
        "human",
        "vehicle",
        "cooking",
        "generic",
        "landscape",
        "space",
        "appearance",
    ]:
        r = random.random()
        if r < 0.4:
            return change_count_inline(caption)
        if r < 0.7:
            return change_existing_color_inline(caption)
        if r < 0.9:
            return insert_object_inline(caption)
        return change_weather_inline(caption)

    # default fallback
    return change_existing_color_inline(caption)


# ---------------------------
# MAIN
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Build augmented prompts + edited contradictions")
    ap.add_argument(
        "--dataset_json",
        required=True,
        help="Input JSON list with {file_name, caption, (optional) type}",
    )
    ap.add_argument("--out_aug", default="prompts_augmented.json")
    ap.add_argument("--out_contra", default="prompts_contradict.json")
    ap.add_argument(
        "--skip_types",
        default="abstract",
        help="Comma-separated item['type'] values to skip",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.dataset_json, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list.")

    skip_set = set(
        [s.strip().lower() for s in (args.skip_types or "").split(",") if s.strip()]
    )

    aug_out: List[Dict[str, Any]] = []
    contra_out: List[Dict[str, Any]] = []

    for idx, it in enumerate(data):
        fn = (it.get("file_name") or "").strip()
        cap = (it.get("caption") or "").strip()
        typ = (it.get("type") or "").strip().lower()
        if not fn or not cap:
            continue
        if typ in skip_set:
            continue

        cat = categorize(cap)

        # augmentation (adds info; can be multi-sentence)
        aug_cap, aug_type = build_best_aug(cap, cat)
        aug_out.append(
            {
                "id": idx,
                "file_name": fn,
                "type": typ,
                "caption": cap,
                "caption_aug": aug_cap,
                "augment_type": aug_type,
                "category": cat,
            }
        )

        # contradiction (DIRECT in-sentence edits only)
        contra_cap, contra_type = build_contradiction(cap, cat)
        contra_out.append(
            {
                "id": idx,
                "file_name": fn,
                "type": typ,
                "caption": cap,
                "caption_contra": contra_cap,
                "contradiction_type": contra_type,
                "category": cat,
            }
        )

    with open(args.out_aug, "w") as f:
        json.dump(aug_out, f, indent=2, ensure_ascii=False)
    with open(args.out_contra, "w") as f:
        json.dump(contra_out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Augmented: {len(aug_out)} -> {args.out_aug}")
    print(f"[OK] Contradictions: {len(contra_out)} -> {args.out_contra}")


if __name__ == "__main__":
    main()

# python utils/prompt_augmenter.py \
#   --dataset_json /capstor/store/cscs/swissai/a144/mariam/vbench2_beta_i2v/data/i2v-bench-info.json \
#   --out_aug prompts_augmented.json \
#   --out_contra prompts_contradict.json \
#   --skip_types abstract \
#   --seed 1234
