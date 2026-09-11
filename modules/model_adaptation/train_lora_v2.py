"""
LoRA training loop v2 — improved over train_lora.py:
  - Uses the answer-balanced training set (fixes 'always guesses no/a' bias)
  - Lower learning rate (2e-5 instead of 1e-4) for more careful learning
  - Early stopping using val_sample.parquet — stops automatically once
    validation loss stops improving, saves the BEST checkpoint separately
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

TRAIN_PATH = DATA_DIR / "train_sample_balanced.parquet"
VAL_PATH = DATA_DIR / "val_sample.parquet"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 2e-5          # lower than v1's 1e-4 — more careful learning
EPOCHS = 3          # upper bound; early stopping will usually cut this short
SAVE_EVERY = 200
EVAL_EVERY = 300    # check validation loss every N optimizer steps
PATIENCE = 3        # stop after this many evals with no improvement
VAL_SUBSET_SIZE = 150  # how many val rows to check each eval (full 960 would be slow)


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
    out = {}
    for k, v in inputs.items():
        v = v.to(device)
        if torch.is_floating_point(v):
            v = v.to(dtype)
        out[k] = v
    return out


@torch.no_grad()
def compute_val_loss(model, val_loader, device, dtype, max_batches):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, inputs in enumerate(val_loader):
        if i >= max_batches:
            break
        inputs = move_to_device(inputs, device, dtype)
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def main():
    print("Loading model + LoRA (bfloat16)...")
    model = get_lora_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")

    print(f"Loading training data from: {TRAIN_PATH}")
    train_ds = BigEarthNetTrainDataset(TRAIN_PATH)
    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    print(f"Loading validation data from: {VAL_PATH}")
    val_ds = BigEarthNetTrainDataset(VAL_PATH)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    step = 0
    best_val_loss = float("inf")
    evals_without_improvement = 0
    optimizer.zero_grad()

    stop_training = False
    for epoch in range(EPOCHS):
        if stop_training:
            break
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

            if step > 0 and step % EVAL_EVERY == 0:
                val_loss = compute_val_loss(model, val_loader, device, torch.bfloat16, VAL_SUBSET_SIZE)
                print(f"  >> VALIDATION at step {step}: val_loss={val_loss:.4f} "
                      f"(best so far: {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    best_path = CKPT_DIR / "best"
                    model.save_pretrained(best_path)
                    print(f"  >> New best. Saved to {best_path}")
                else:
                    evals_without_improvement += 1
                    print(f"  >> No improvement ({evals_without_improvement}/{PATIENCE})")
                    if evals_without_improvement >= PATIENCE:
                        print("  >> Early stopping triggered.")
                        stop_training = True
                        break

            step += 1

    final_path = CKPT_DIR / "final_v2"
    model.save_pretrained(final_path)
    print(f"Training complete. Final adapter saved to {final_path}")
    print(f"Best adapter (lowest val loss) saved to {CKPT_DIR / 'best'}")


if __name__ == "__main__":
    main()
