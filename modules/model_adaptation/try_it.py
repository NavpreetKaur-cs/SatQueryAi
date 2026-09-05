"""
Quick manual test: point at any image OR a BigEarthNet patch_id, ask any question.

Usage (regular photo):
    python -m modules.model_adaptation.try_it --image /path/to/photo.jpg "your question"

Usage (BigEarthNet satellite patch):
    python -m modules.model_adaptation.try_it --patch <patch_id> "your question"
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from modules.model_adaptation.infer import RemoteSensingVLM

DATA_DIR = Path.home() / "datasets" / "BigEarthNet.txt"
SUBSET_DIR = DATA_DIR / "bigearthnet_subset" / "S2"
RGB_BANDS = ["B04", "B03", "B02"]


def load_rgb_patch(patch_id):
    patch_dir = SUBSET_DIR / patch_id
    bands = []
    for b in RGB_BANDS:
        with rasterio.open(patch_dir / f"{patch_id}_{b}.tif") as src:
            bands.append(src.read(1).astype(np.float32))
    rgb = np.stack(bands, axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a regular image file (jpg/png)")
    group.add_argument("--patch", help="BigEarthNet patch_id (builds RGB from S2 bands)")
    parser.add_argument("question", help="The question to ask about the image")
    args = parser.parse_args()

    print("Loading model...")
    vlm = RemoteSensingVLM()

    if args.image:
        print(f"Loading image: {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        print(f"Building RGB from patch: {args.patch}")
        image = load_rgb_patch(args.patch)

    print(f"Question: {args.question}")
    result = vlm.infer(image=image, query=args.question)

    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
