import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import urllib.request

test_img_path = "/tmp/test_bee.jpg"
urllib.request.urlretrieve(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    test_img_path,
)

model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(
    model_checkpoint, torch_dtype=torch.float32, device_map="auto"
)

image = Image.open(test_img_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=50)
decoded = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\nOutput:", decoded)
