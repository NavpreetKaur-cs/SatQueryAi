"""
PyTorch Dataset that loads real BigEarthNet patches from bigearthnet_subset/
and formats them for InternVL3-1B-hf training.
"""
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
from torch.utils.data import Dataset

SUBSET_DIR = Path.home() / "datasets" / "BigEarthNet.txt" / "bigearthnet_subset" / "S2"
RGB_BANDS = ["B04", "B03", "B02"]  # Red, Green, Blue


def load_rgb_patch(patch_id: str) -> Image.Image:
    """Load B04/B03/B02 bands from a patch folder and compose an RGB PIL image."""
    patch_dir = SUBSET_DIR / patch_id
    bands = []
    for b in RGB_BANDS:
        tif_path = patch_dir / f"{patch_id}_{b}.tif"
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        bands.append(arr)

    rgb = np.stack(bands, axis=-1)
    # simple percentile stretch for visibility (satellite reflectance isn't 0-255 by default)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


class BigEarthNetTrainDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        available = {p.name for p in SUBSET_DIR.iterdir() if p.is_dir()}
        self.df = df[df["patch_id"].isin(available)].reset_index(drop=True)
        print(f"Dataset: {len(df)} rows in parquet, {len(self.df)} usable (image available)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_rgb_patch(row["patch_id"])
        return {
            "image": image,
            "question": row["input"],
            "answer": str(row["output"]),
        }


if __name__ == "__main__":
    ds = BigEarthNetTrainDataset(
        Path.home() / "datasets" / "BigEarthNet.txt" / "train_sample.parquet"
    )
    print(f"\nUsable training examples: {len(ds)}")
    sample = ds[0]
    print(f"\nSample question: {sample['question']}")
    print(f"Sample answer: {sample['answer']}")
    print(f"Image size: {sample['image'].size}, mode: {sample['image'].mode}")
    sample["image"].save("/tmp/sample_patch.png")
    print("Saved preview to /tmp/sample_patch.png")
