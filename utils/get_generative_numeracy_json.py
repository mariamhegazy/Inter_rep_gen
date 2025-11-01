#!/usr/bin/env python3
import json
import re

# -------- CONFIG --------
input_json = "/capstor/store/cscs/swissai/a144/mariam/T2V_compbench_images/json_outputs/Two_animals_experiments/two_animals_fight_full_prompt/sample_0000.json"
output_json = "meta_data_animals/two_animals_fight_generative_numeracy.json"
# ------------------------

# Mapping number words to digits
NUMBER_MAP = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

# Regex to extract animal(s)
ANIMAL_REGEX = re.compile(
    r"Two anthropomorphic ([a-zA-Z_ ]+?)s? in comfy boxing gear", re.IGNORECASE
)

with open(input_json, "r") as f:
    data = json.load(f)

formatted = []

for entry in data:
    caption = entry["caption"]
    lower_caption = caption.lower()

    # --- extract number word ---
    number_word = None
    for word in NUMBER_MAP.keys():
        if word in lower_caption:
            number_word = word
            break
    number_value = NUMBER_MAP.get(number_word, "")

    # --- extract animal/object name ---
    m = ANIMAL_REGEX.search(caption)
    if m:
        obj = m.group(1).strip()
        # handle plural → singular manually
        obj = obj.replace("anthropomorphic ", "").strip()
    else:
        obj = "unknown"

    formatted.append({"caption": caption, "objects": obj, "numbers": number_value})

with open(output_json, "w") as f:
    json.dump(formatted, f, indent=2)

print(f"✅ Saved formatted JSON to: {output_json}")
print(f"Converted {len(formatted)} entries.")
