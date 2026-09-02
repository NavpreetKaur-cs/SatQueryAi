import pandas as pd
from pathlib import Path

PARQUET_PATH = Path.home() / "datasets" / "BigEarthNet.txt" / "BigEarthNet.txt.parquet"
OUT_DIR = Path("data/bigearthnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VQA_CATEGORIES = ["presence", "area", "count", "adjacency", "season", "climate zone", "country"]
TRAIN_PER_CATEGORY = 1000
VAL_PER_CATEGORY = 120
RANDOM_STATE = 42

def stratified_sample(df, split_name, per_category):
    split_df = df[df["split"] == split_name]
    parts = []

    vqa = split_df[split_df["type"].isin(["binary", "mcq"]) & split_df["category"].isin(VQA_CATEGORIES)]
    for cat in VQA_CATEGORIES:
        cat_df = vqa[vqa["category"] == cat]
        n = min(per_category, len(cat_df))
        parts.append(cat_df.sample(n, random_state=RANDOM_STATE))

    cap = split_df[split_df["type"] == "captioning"]
    n_cap = min(per_category, len(cap))
    parts.append(cap.sample(n_cap, random_state=RANDOM_STATE))

    return pd.concat(parts, ignore_index=True)

def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    print("Sampling train...")
    train_sample = stratified_sample(df, "train", TRAIN_PER_CATEGORY)
    print("Sampling validation...")
    val_sample = stratified_sample(df, "validation", VAL_PER_CATEGORY)

    print("Taking full bench split (held-out eval, untouched)...")
    bench_sample = df[df["split"] == "bench"].reset_index(drop=True)

    print(f"\nTrain sample: {len(train_sample)} rows")
    print(train_sample["type"].value_counts())
    print(f"\nVal sample: {len(val_sample)} rows")
    print(val_sample["type"].value_counts())
    print(f"\nBench (full, held-out): {len(bench_sample)} rows")

    train_sample.to_parquet(OUT_DIR / "train_sample.parquet")
    val_sample.to_parquet(OUT_DIR / "val_sample.parquet")
    bench_sample.to_parquet(OUT_DIR / "bench_full.parquet")

    all_samples = pd.concat([train_sample, val_sample, bench_sample])
    unique_patch_ids = sorted(all_samples["patch_id"].unique())
    unique_s1_names = sorted(all_samples["s1_name"].unique())

    (OUT_DIR / "needed_patch_ids.txt").write_text("\n".join(unique_patch_ids))
    (OUT_DIR / "needed_s1_names.txt").write_text("\n".join(unique_s1_names))

    print(f"\nUnique S2 patch_ids needed: {len(unique_patch_ids)}")
    print(f"Unique S1 names needed: {len(unique_s1_names)}")

if __name__ == "__main__":
    main()
