import csv
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

BASE = Path.home() / "datasets" / "BigEarthNet.txt"
S2_DIR = BASE / "bigearthnet_subset" / "S2"
S1_DIR = BASE / "bigearthnet_subset" / "S1"
OUT_DIR = Path("data/optical_sar")
(OUT_DIR / "optical").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "sar").mkdir(parents=True, exist_ok=True)

KEYWORDS = {
    "built_up": ["urban fabric", "industrial or commercial"],
    "water": ["inland waters", "marine waters", "inland wetlands", "coastal wetlands"],
    "vegetation": ["forest", "grassland", "moors", "heathland", "sclerophyllous",
                   "transitional woodland", "agro-forestry", "shrub"],
}

def derive_labels(questions_df):
    labels = {"built_up": 0, "water": 0, "vegetation": 0}
    for _, row in questions_df.iterrows():
        q = row["input"].lower()
        ans = str(row["output"]).strip().lower()
        if ans != "yes":
            continue
        for cat, kws in KEYWORDS.items():
            if any(kw in q for kw in kws):
                labels[cat] = 1
    return labels

def build_composite_rgb(patch_id, out_path):
    bands = ["B04", "B03", "B02"]
    patch_dir = S2_DIR / patch_id
    arrs, profile = [], None
    for b in bands:
        with rasterio.open(patch_dir / f"{patch_id}_{b}.tif") as src:
            arrs.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    profile.update(count=3, dtype="float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, arr in enumerate(arrs, start=1):
            dst.write(arr.astype("float32"), i)

def main():
    print("Loading BigEarthNet.txt metadata...")
    df = pd.read_parquet(BASE / "BigEarthNet.txt.parquet")
    presence_df = df[(df["type"] == "binary") & (df["category"] == "presence")]

    s2_patches = {p.name for p in S2_DIR.iterdir() if p.is_dir()}
    s1_lookup = {p.name for p in S1_DIR.iterdir() if p.is_dir()}
    patch_to_s1 = dict(zip(df["patch_id"], df["s1_name"]))

    rows_out = []
    skipped = 0
    for patch_id in sorted(s2_patches):
        s1_name = patch_to_s1.get(patch_id)
        if not s1_name or s1_name not in s1_lookup:
            skipped += 1
            continue

        vv_path = S1_DIR / s1_name / f"{s1_name}_VV.tif"
        if not vv_path.exists():
            skipped += 1
            continue

        questions = presence_df[presence_df["patch_id"] == patch_id]
        if len(questions) == 0:
            skipped += 1
            continue
        labels = derive_labels(questions)

        optical_out = OUT_DIR / "optical" / f"{patch_id}.tif"
        sar_out = OUT_DIR / "sar" / f"{s1_name}.tif"

        try:
            if not optical_out.exists():
                build_composite_rgb(patch_id, optical_out)
            if not sar_out.exists():
                with rasterio.open(vv_path) as src:
                    profile = src.profile.copy()
                    data = src.read(1)
                with rasterio.open(sar_out, "w", **profile) as dst:
                    dst.write(data, 1)
        except Exception as e:
            print(f"  [skip] {patch_id}: {e}")
            skipped += 1
            continue

        rows_out.append({
            "optical_path": f"optical/{patch_id}.tif",
            "sar_path": f"sar/{s1_name}.tif",
            "built_up": labels["built_up"],
            "water": labels["water"],
            "vegetation": labels["vegetation"],
        })

        if len(rows_out) % 200 == 0:
            print(f"  Built {len(rows_out)} pairs...")

    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optical_path", "sar_path", "built_up", "water", "vegetation"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nDone. {len(rows_out)} usable pairs written, {skipped} skipped.")
    print(f"Label distribution: built_up={sum(r['built_up'] for r in rows_out)}, "
          f"water={sum(r['water'] for r in rows_out)}, "
          f"vegetation={sum(r['vegetation'] for r in rows_out)}")

if __name__ == "__main__":
    main()
