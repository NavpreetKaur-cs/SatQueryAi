"""
Evaluates InternVL3-1B-hf (base vs LoRA-adapted) on the bench split.
Uses proper answer extraction instead of substring matching.
"""
import re
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

DATA_DIR = Path.home() / "datasets" / "BigEarthNet.txt"
SUBSET_DIR = DATA_DIR / "bigearthnet_subset" / "S2"
RGB_BANDS = ["B04", "B03", "B02"]
CKPT_PATH = Path("models/adaptation/checkpoints/final")
N_EVAL = 100


def load_rgb_patch(patch_id):
    patch_dir = SUBSET_DIR / patch_id
    bands = []
    for b in RGB_BANDS:
        with rasterio.open(patch_dir / f"{patch_id}_{b}.tif") as src:
            bands.append(src.read(1).astype(np.float32))
    rgb = np.stack(bands, axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def extract_answer(pred, qtype):
    pred = pred.strip().lower()
    if qtype == "mcq":
        m = re.match(r"^\s*([abcd])\b", pred)
        return m.group(1) if m else pred[:1]
    if qtype == "binary":
        if pred.startswith("yes"):
            return "yes"
        if pred.startswith("no"):
            return "no"
        return pred
    return pred


def generate_answer(model, processor, image, question, device):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_eval(model, processor, df, device, label):
    correct, total, examples = 0, 0, []
    for _, row in df.iterrows():
        patch_dir = SUBSET_DIR / row["patch_id"]
        if not patch_dir.exists():
            continue
        image = load_rgb_patch(row["patch_id"])
        pred = generate_answer(model, processor, image, row["input"], device)
        gt = str(row["output"]).strip().lower()
        if row["type"] in ("binary", "mcq"):
            pred_answer = extract_answer(pred, row["type"])
            is_correct = pred_answer == gt
            total += 1
            correct += is_correct
        examples.append((row["type"], row["input"][:60], gt, pred[:40]))
    acc = correct / total if total else 0.0
    print(f"\n=== {label} === accuracy on binary/mcq: {correct}/{total} = {acc:.2%}")
    for t, q, gt, pred in examples[:8]:
        print(f"  [{t}] Q: {q}... GT: {gt!r} PRED_RAW: {pred!r}")
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")

    bench = pd.read_parquet(DATA_DIR / "bench.parquet")
    available = {p.name for p in SUBSET_DIR.iterdir() if p.is_dir()}
    bench = bench[bench["patch_id"].isin(available)].reset_index(drop=True)
    bench_sample = bench.sample(min(N_EVAL, len(bench)), random_state=42)
    print(f"Evaluating on {len(bench_sample)} bench examples (of {len(bench)} available)")

    print("\nLoading BASE model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()
    with torch.no_grad():
        base_acc = run_eval(base_model, processor, bench_sample, device, "BASE (unadapted)")

    del base_model
    torch.cuda.empty_cache()

    print("\nLoading ADAPTED model (base + LoRA)...")
    adapted_base = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16
    ).to(device)
    adapted_model = PeftModel.from_pretrained(adapted_base, CKPT_PATH)
    adapted_model.eval()
    with torch.no_grad():
        adapted_acc = run_eval(adapted_model, processor, bench_sample, device, "ADAPTED (LoRA fine-tuned)")

    print(f"\n{'='*50}")
    print(f"RESULTS: base={base_acc:.2%}  adapted={adapted_acc:.2%}  delta={adapted_acc-base_acc:+.2%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
