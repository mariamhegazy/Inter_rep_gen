import json
import os
import random

# Folder containing your 16 JSON files
input_dir = "/path/to/json_folder"
output_dir = os.path.join(input_dir, "with_after_phrases")
os.makedirs(output_dir, exist_ok=True)

# You can extend this list with anything you want
after_phrases = [
    "after a ball",
    "after a frisbee",
    "after a rabit",
]

# Iterate over all JSON files in the folder
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    with open(input_path, "r") as f:
        data = json.load(f)

    for item in data:
        phrase = random.choice(after_phrases)
        # ensure only one period at end
        caption = item["caption"].rstrip(".")
        item["caption"] = f"{caption} after {phrase}."

    # Save to output folder
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[done] {filename} → {output_path}")

print("✅ All JSON files updated and saved to:", output_dir)
