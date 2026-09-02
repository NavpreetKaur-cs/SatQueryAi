import pandas as pd
from pathlib import Path

PARQUET_PATH = Path.home() / "datasets" / "BigEarthNet.txt" / "BigEarthNet.txt.parquet"

def main():
    print(f"Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\n--- Rows per split ---")
    print(df["split"].value_counts())

    print("\n--- Rows per type ---")
    print(df["type"].value_counts())

    print("\n--- Rows per category ---")
    print(df["category"].value_counts())

    print("\n--- 3 sample rows per type ---")
    for t in df["type"].unique():
        subset = df[df["type"] == t]
        sample = subset.sample(min(3, len(subset)), random_state=42)
        print(f"\n=== type: {t} ===")
        for _, row in sample.iterrows():
            print(f"  s1_name:  {row['s1_name']}")
            print(f"  patch_id: {row['patch_id']}")
            print(f"  input:    {row['input']}")
            print(f"  output:   {row['output']}")
            print(f"  category: {row['category']}, split: {row['split']}")
            print()

if __name__ == "__main__":
    main()