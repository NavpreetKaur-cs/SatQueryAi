"""
modules/multimodal/training/extract_features_cache.py

One-time pass over a labeled optical/SAR dataset: loads each pair,
preprocesses, extracts features with the frozen backbones, and caches
(feature_vector, labels) to disk so train_task_head.py never has to
re-run the CNN forward pass.

Expected dataset layout (adjust `load_dataset_index` if yours differs):

    data/optical_sar/
        index.csv           # columns: optical_path, sar_path, built_up, water, vegetation
        optical/...          # referenced by optical_path
        sar/...               # referenced by sar_path

`built_up`, `water`, `vegetation` should be 0/1 presence labels.

Run from the repo root:
    python -m modules.multimodal.training.extract_features_cache
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from modules.multimodal.preprocessing import load_and_validate_pair, CoRegistrationError
from modules.multimodal.feature_extraction import MultimodalFeatureExtractor
from modules.multimodal.fusion import MultimodalFusion

CATEGORY_ORDER = ["built_up", "water", "vegetation"]


def load_dataset_index(index_path: Path):
    """
    Reads index.csv and returns a list of dicts:
        {"optical_path": str, "sar_path": str, "labels": [0/1, 0/1, 0/1]}
    """
    rows = []
    with open(index_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = [int(row[cat]) for cat in CATEGORY_ORDER]
            rows.append({
                "optical_path": row["optical_path"],
                "sar_path": row["sar_path"],
                "labels": labels,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="data/optical_sar",
        help="Directory containing index.csv and the image pairs.",
    )
    parser.add_argument(
        "--output", type=str, default="models/multimodal/checkpoints/features_cache.npz",
        help="Where to save the cached features + labels.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_path = data_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Expected an index file at {index_path}. See this script's "
            "docstring for the expected format."
        )

    rows = load_dataset_index(index_path)
    print(f"Loaded {len(rows)} labeled pairs from {index_path}")

    extractor = MultimodalFeatureExtractor(optical_channels=3, sar_channels=1, pretrained=False)
    fusion = MultimodalFusion(strategy="concat", feature_dim=512)

    fused_vectors = []
    labels = []
    skipped = 0

    for i, row in enumerate(rows):
        optical_path = str(data_dir / row["optical_path"])
        sar_path = str(data_dir / row["sar_path"])

        try:
            optical_arr, sar_arr = load_and_validate_pair(optical_path, sar_path)
            optical_feat, sar_feat = extractor.extract(optical_arr, sar_arr)
            fused = fusion.fuse(optical_feat.vector, sar_feat.vector)
        except (CoRegistrationError, Exception) as e:
            print(f"  [skip] {optical_path} / {sar_path}: {e}")
            skipped += 1
            continue

        fused_vectors.append(fused)
        labels.append(row["labels"])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rows)}")

    print(f"Done. {len(fused_vectors)} usable pairs, {skipped} skipped.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=np.stack(fused_vectors),
        labels=np.array(labels, dtype=np.float32),
        category_order=np.array(CATEGORY_ORDER),
    )
    print(f"Saved feature cache to {output_path}")


if __name__ == "__main__":
    main()