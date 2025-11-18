#!/usr/bin/env python

import glob
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def get_first_frame(video_path):
    """Return first frame as a PIL Image using decord."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) == 0:
            return None
        frame = vr[0].asnumpy()  # (H, W, 3), RGB
        return Image.fromarray(frame)
    except Exception as e:
        print(f"  Error reading {video_path} with decord: {e}")
        return None


def main(video_dir, output_csv="clipscores.csv", device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    exts = ("*.mp4", "*.webm", "*.avi", "*.mov", "*.mkv")
    video_paths = []
    for e in exts:
        video_paths.extend(glob.glob(os.path.join(video_dir, e)))

    video_paths = sorted(video_paths)
    if not video_paths:
        print("No videos found.")
        return

    results = []

    for vp in video_paths:
        fname = os.path.basename(vp)
        prompt = os.path.splitext(fname)[0].replace("_", " ").replace("-", " ")

        print(f"Processing: {fname} | prompt: '{prompt}'")

        frame = get_first_frame(vp)
        if frame is None:
            print(f"  Could not read first frame, skipping.")
            continue

        inputs = processor(
            text=[prompt], images=[frame], return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds  # (1, D)
            txt_emb = outputs.text_embeds  # (1, D)
            score = F.cosine_similarity(img_emb, txt_emb, dim=-1).item()

        results.append(
            {
                "video_path": vp,
                "video_name": fname,
                "prompt": prompt,
                "clip_cosine": score,
            }
        )

    if not results:
        print("No CLIP scores computed (all videos failed to decode?).")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} scores to {output_csv}")

    # Compute and print average CLIP score
    avg_score = df["clip_cosine"].mean()
    print(f"Average CLIP cosine score over {len(df)} videos: {avg_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python compute_clipscore_first_frame_decord.py <video_dir> [output_csv]"
        )
        sys.exit(1)

    video_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "clipscores.csv"
    main(video_dir, output_csv)
