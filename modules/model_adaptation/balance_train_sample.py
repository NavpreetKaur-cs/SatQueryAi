"""
Balance the LoRA training sample so 'yes'/'no' and mcq letters (a/b/c/d...)
appear roughly equally often. This reduces the model's tendency to just
guess the most common answer instead of actually reading the image.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "datasets" / "BigEarthNet.txt"
IN_PATH = DATA_DIR / "train_sample.parquet"
OUT_PATH = DATA_DIR / "train_sample_balanced.parquet"

ANSWER_COL = "output"  # <-- change this if your column has a different name

df = pd.read_parquet(IN_PATH)
print("Columns:", list(df.columns))
print("Total rows before balancing:", len(df))

if ANSWER_COL not in df.columns:
    raise SystemExit(
        f"'{ANSWER_COL}' not found in columns above. "
        f"Edit ANSWER_COL at the top of this script to the correct column name and re-run."
    )

def extract_answer_key(ans):
    s = str(ans).strip().lower()
    if s in ("yes", "no"):
        return s
    if len(s) == 1 and s.isalpha():
        return s
    return "other"

df["_answer_key"] = df[ANSWER_COL].apply(extract_answer_key)
print("\nAnswer distribution BEFORE balancing:")
print(df["_answer_key"].value_counts())

is_bucketed = df["_answer_key"] != "other"
bucketed = df[is_bucketed]
other = df[~is_bucketed]

if len(bucketed) == 0:
    raise SystemExit("No binary/mcq rows detected — check ANSWER_COL and extract_answer_key().")

max_count = bucketed["_answer_key"].value_counts().max()
print(f"\nOversampling every binary/mcq answer type up to {max_count} rows each "
      f"(duplicates rare answers, does not delete anything).")

balanced_parts = []
for key, group in bucketed.groupby("_answer_key"):
    replace = len(group) < max_count
    balanced_parts.append(group.sample(n=max_count, replace=replace, random_state=42))

balanced = pd.concat(balanced_parts + [other]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced = balanced.drop(columns=["_answer_key"])

print(f"\nBalanced dataset: {len(balanced)} rows (was {len(df)})")
balanced.to_parquet(OUT_PATH)
print(f"Saved to: {OUT_PATH}")
print("\nNext: point train_lora.py at this file instead of train_sample.parquet.")
