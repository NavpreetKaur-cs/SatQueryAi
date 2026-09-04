"""
Clean inference interface for the remote-sensing-adapted VLM.
This is what the agent module (agent/executor.py) should call.

Usage:
    from modules.model_adaptation.infer import RemoteSensingVLM
    vlm = RemoteSensingVLM()
    result = vlm.infer(image="path/to/patch_or_PIL_image", query="Does this image show forest?")
"""
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID = "OpenGVLab/InternVL3-1B-hf"
DEFAULT_ADAPTER_PATH = Path(__file__).parent.parent.parent / "models" / "adaptation" / "checkpoints" / "final"


class RemoteSensingVLM:
    def __init__(self, adapter_path=DEFAULT_ADAPTER_PATH, device=None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.adapter_path = Path(adapter_path)

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        base_model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=self.dtype)

        if self.adapter_path.exists():
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.adapted = True
        else:
            self.model = base_model
            self.adapted = False

        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image):
        """Accepts a file path (str/Path) or an already-loaded PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def infer(self, image, query, max_new_tokens=50):
        """
        Run inference on a single image + natural-language query.

        Returns a dict matching the team's shared I/O contract:
            success: bool
            answer: str (empty on failure)
            confidence: float or None (not computed by this model yet)
            visual_output: None (this model does not produce visual outputs)
            model: str (which model/adapter answered)
            error: str or None
        """
        try:
            pil_image = self._load_image(image)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query},
                ],
            }]

            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            inputs = {
                k: (v.to(self.dtype) if torch.is_floating_point(v) else v)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            answer = self.processor.decode(
                out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            return {
                "success": True,
                "answer": answer,
                "confidence": None,
                "visual_output": None,
                "model": f"{MODEL_ID}" + ("+LoRA" if self.adapted else " (base, no adapter found)"),
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "answer": "",
                "confidence": None,
                "visual_output": None,
                "model": MODEL_ID,
                "error": str(e),
            }


if __name__ == "__main__":
    # Smoke test using a real patch from the subset, if available
    import pandas as pd

    data_dir = Path.home() / "datasets" / "BigEarthNet.txt"
    subset_dir = data_dir / "bigearthnet_subset" / "S2"

    vlm = RemoteSensingVLM()
    print(f"Loaded model. Adapter found: {vlm.adapted}")

    # find any one available patch to test with
    sample_patch = next(subset_dir.iterdir())
    tif_files = list(sample_patch.glob("*_B04.tif"))
    if tif_files:
        import rasterio
        import numpy as np

        patch_id = sample_patch.name
        bands = []
        for b in ["B04", "B03", "B02"]:
            with rasterio.open(sample_patch / f"{patch_id}_{b}.tif") as src:
                bands.append(src.read(1).astype(np.float32))
        rgb = np.stack(bands, axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        test_image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

        result = vlm.infer(test_image, "Describe the land cover shown in this image.")
        print("\nTest result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("No sample patch found for smoke test.")
