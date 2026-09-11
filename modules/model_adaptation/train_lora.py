"""
LoRA training loop for InternVL3-1B-hf on real BigEarthNet.txt data.
Memory-optimized for 8GB VRAM: bfloat16 + gradient checkpointing.
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from transformers import AutoProcessor
from lora_config import get_lora_model
from build_training_dataset import BigEarthNetTrainDataset

DATA_DIR = Path.home() / "datasets" / "BigEarthNet.txt"
CKPT_DIR = Path("models/adaptation/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 1e-4
EPOCHS = 1
SAVE_EVERY = 200


def collate_fn(batch, processor):
    item = batch[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": item["answer"]}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def move_to_device(inputs, device, dtype):
    """Cast float tensors (image pixels) to model dtype; keep integer tensors (ids/masks) as-is."""
    out = {}
    for k, v in inputs.items():
        v = v.to(device)
        if torch.is_floating_point(v):
            v = v.to(dtype)
        out[k] = v
    return out


def main():
    print("Loading model + LoRA (bfloat16)...")
    model = get_lora_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")

    print("Loading dataset...")
    train_ds = BigEarthNetTrainDataset(DATA_DIR / "train_sample.parquet")

    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    step = 0
    optimizer.zero_grad()
    for epoch in range(EPOCHS):
        for inputs in loader:
            inputs = move_to_device(inputs, device, torch.bfloat16)
            outputs = model(**inputs)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item() * GRAD_ACCUM_STEPS:.4f}")

            if step > 0 and step % SAVE_EVERY == 0:
                save_path = CKPT_DIR / f"step_{step}"
                model.save_pretrained(save_path)
                print(f"Saved checkpoint: {save_path}")

            step += 1

    final_path = CKPT_DIR / "final"
    model.save_pretrained(final_path)
    print(f"Training complete. Final adapter saved to {final_path}")


if __name__ == "__main__":
    main()
