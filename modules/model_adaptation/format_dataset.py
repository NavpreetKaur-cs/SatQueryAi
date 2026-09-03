"""
Converts BigEarthNet.txt sample rows into InternVL3-1B-hf's expected
chat-message format for training/inference.

This does NOT load images yet - it only builds the text-side structure,
so it can be tested and verified before the data pipeline finishes.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/bigearthnet")


def row_to_messages(row: pd.Series) -> list[dict]:
    """
    Convert one BigEarthNet.txt row into InternVL chat-format messages.

    Matches the format verified working in test_internvl_local.py:
    content is a list of {"type": "image"/"text", ...} dicts.
    """
    question = row["input"]
    answer = str(row["output"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},  # filled in later with real PIL image
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]
    return messages


def main():
    train_df = pd.read_parquet(DATA_DIR / "train_sample.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_sample.parquet")

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")

    # Sanity-check on a few real rows, across different types
    print("\n--- Example formatted messages ---")
    for t in train_df["type"].unique():
        sample_row = train_df[train_df["type"] == t].iloc[0]
        msgs = row_to_messages(sample_row)
        print(f"\ntype={t}, category={sample_row['category']}")
        print(f"  patch_id: {sample_row['patch_id']}")
        print(f"  question: {msgs[0]['content'][1]['text']}")
        print(f"  answer:   {msgs[1]['content'][0]['text']}")


if __name__ == "__main__":
    main()
