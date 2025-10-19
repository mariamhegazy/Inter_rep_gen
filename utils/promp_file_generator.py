import json
from pathlib import Path

VIDEOS_DIR = Path(
    "/capstor/scratch/cscs/mhasan/VideoX-Fun/samples/wan-videos-vbench-ours/T2V/AUG"
)  # set this
USE_AUG = True  # True if you want caption_aug

data = json.load(open("prompts_augmented.json"))  # your array of objects
out = {}

for item in data:
    base = Path(item["file_name"]).stem  # strip .jpg
    video_path = str(VIDEOS_DIR / f"{base}.mp4")  # or .webm
    prompt = item["caption_aug"] if USE_AUG else item["caption"]
    out[video_path] = prompt

json.dump(out, open("prompts_aug_T2V.json", "w"), indent=2)
print("Wrote prompts.json with", len(out), "entries")


# VIDEOS_DIR = Path(
#     "/capstor/scratch/cscs/mhasan/VideoX-Fun/samples/wan-videos-vbench/I2V/"
# )  # set this
# USE_AUG = False  # True if you want caption_aug

# data = json.load(open("prompts_contradict.json"))  # your array of objects
# out = {}

# for item in data:
#     base = Path(item["file_name"]).stem  # strip .jpg
#     video_path = str(VIDEOS_DIR / f"{base}.mp4")  # or .webm
#     prompt = item["caption_contra"] if USE_AUG else item["caption"]
#     out[video_path] = prompt

# json.dump(out, open("prompts_base_I2V.json", "w"), indent=2)
# print("Wrote prompts.json with", len(out), "entries")
