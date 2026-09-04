"""
Final evaluation: BASE vs ADAPTED V1 vs ADAPTED V2 on 2000 bench questions,
with per-category accuracy breakdown.
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
CKPT_V1 = Path("models/adaptation/checkpoints/final")
CKPT_V2 = Path("models/adaptation/checkpoints/best")
N_EVAL = 2000


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
    correct, total = 0, 0
    answer_counts = {}
    cat_correct = {}
    cat_total = {}
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
            answer_counts[pred_answer] = answer_counts.get(pred_answer, 0) + 1
            cat = row.get("category", "unknown")
            cat_total[cat] = cat_total.get(cat, 0) + 1
            cat_correct[cat] = cat_correct.get(cat, 0) + is_correct
    acc = correct / total if total else 0.0
    print(f"\n=== {label} === accuracy on binary/mcq: {correct}/{total} = {acc:.2%}")
    print(f"  Answer distribution given by model: {answer_counts}")
    print(f"  Per-category accuracy:")
    for cat in sorted(cat_total.keys()):
        c_acc = cat_correct[cat] / cat_total[cat]
        print(f"    {cat}: {cat_correct[cat]}/{cat_total[cat]} = {c_acc:.2%}")
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

    print("\nLoading ADAPTED V1 (original LoRA)...")
    v1_base = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16
    ).to(device)
    v1_model = PeftModel.from_pretrained(v1_base, CKPT_V1)
    v1_model.eval()
    with torch.no_grad():
        v1_acc = run_eval(v1_model, processor, bench_sample, device, "ADAPTED V1 (original)")
    del v1_model, v1_base
    torch.cuda.empty_cache()

    print("\nLoading ADAPTED V2 (balanced + lower LR + early stopping)...")
    v2_base = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16
    ).to(device)
    v2_model = PeftModel.from_pretrained(v2_base, CKPT_V2)
    v2_model.eval()
    with torch.no_grad():
        v2_acc = run_eval(v2_model, processor, bench_sample, device, "ADAPTED V2 (improved)")

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON (on {len(bench_sample)} questions)")
    print(f"  BASE:       {base_acc:.2%}")
    print(f"  ADAPTED V1: {v1_acc:.2%}  (delta vs base: {v1_acc-base_acc:+.2%})")
    print(f"  ADAPTED V2: {v2_acc:.2%}  (delta vs base: {v2_acc-base_acc:+.2%})")
    print(f"  V2 vs V1:   {v2_acc-v1_acc:+.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
