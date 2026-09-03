import torch
from transformers import pipeline

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

pipe = pipeline(
    "image-text-to-text",
    model="OpenGVLab/InternVL3-1B-hf",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float32,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    },
]

outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
print("\nModel output:")
print(outputs[0]["generated_text"])
