"""
LoRA configuration for InternVL3-1B-hf, sized for 8GB VRAM.
Target modules verified from actual model introspection (inspect_model_layers.py).
"""
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForImageTextToText

def get_lora_model():
    model = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.float32
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

if __name__ == "__main__":
    model = get_lora_model()
