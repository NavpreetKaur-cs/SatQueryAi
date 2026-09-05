import torch

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


MODEL_ID = "OpenGVLab/InternVL3-1B-hf"


def main():
    print("=== Environment ===")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )

    print("\n=== Loading processor ===")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Processor loaded.")

    print("\n=== 4-bit configuration ===")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Quantization config created.")

    print("\n=== Loading model in 4-bit ===")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Model loaded.")

    print("\n=== Preparing model for k-bit training ===")

    model = prepare_model_for_kbit_training(model)

    print("Model prepared.")

    print("\n=== Applying LoRA ===")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    print("LoRA applied.")

    print("\n=== Parameter count ===")

    trainable = 0
    total = 0

    for param in model.parameters():
        total += param.numel()

        if param.requires_grad:
            trainable += param.numel()

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable percentage:  {100 * trainable / total:.4f}%")

    print("\n=== SUCCESS ===")
    print("4-bit InternVL3 + LoRA setup works.")


if __name__ == "__main__":
    main()
