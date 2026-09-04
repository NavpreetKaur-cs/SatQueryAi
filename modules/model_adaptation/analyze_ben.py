import pandas as pd
from pathlib import Path

PATH = Path.home() / "datasets" / "BigEarthNet.txt" / "BigEarthNet.txt.parquet"

df = pd.read_parquet(PATH)

print("=" * 70)
print("BIGEARTHNET-TXT DATASET")
print("=" * 70)

print(f"\nTotal QA rows: {len(df):,}")
print(f"Unique S1 images: {df['s1_name'].nunique():,}")
print(f"Unique S2 patches: {df['patch_id'].nunique():,}")

print("\n--- SPLITS ---")
print(df["split"].value_counts())

print("\n--- TYPES ---")
print(df["type"].value_counts())

print("\n--- CATEGORIES ---")
print(df["category"].value_counts())

print("\n--- QUESTIONS PER PATCH ---")
q_per_patch = df.groupby("patch_id").size()
print(q_per_patch.describe())

print("\n--- COUNTRIES ---")
print(df["country"].value_counts().head(20))

print("\n--- SEASONS ---")
print(df["season"].value_counts())

print("\n--- CLIMATE ZONES ---")
print(df["climate_zone"].value_counts())

print("\n--- EXAMPLE QUESTIONS ---")

for category in sorted(df["category"].dropna().unique()):
    subset = df[df["category"] == category]

    print(f"\n### CATEGORY: {category}")

    for _, row in subset.sample(
        min(5, len(subset)),
        random_state=42
    ).iterrows():

        print("PATCH :", row["patch_id"])
        print("TYPE  :", row["type"])
        print("QUESTION:", row["input"])
        print("ANSWER  :", row["output"])
        print()
