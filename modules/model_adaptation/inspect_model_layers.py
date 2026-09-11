"""
Introspects InternVL3-1B-hf's actual module structure to determine
correct LoRA target_modules - verified from the real model, not guessed.
"""
import torch
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.float32
)

print("=== Top-level structure ===")
for name, _ in model.named_children():
    print(name)

print("\n=== All Linear layer names containing common attention keywords ===")
seen_suffixes = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        suffix = name.split(".")[-1]
        seen_suffixes.add(suffix)

print(sorted(seen_suffixes))

print("\n=== Sample full paths (language model attention layers) ===")
count = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "language_model" in name:
        print(name)
        count += 1
        if count >= 10:
            break

print(f"\nTotal trainable params (full model): {sum(p.numel() for p in model.parameters()):,}")
