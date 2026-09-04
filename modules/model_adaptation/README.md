# Model Adaptation — Person 1

## Model
Base model: **InternVL3-1B-hf** (OpenGVLab/InternVL3-1B-hf), a vision-language model
combining a custom vision encoder with a Qwen2-based 1B-parameter language decoder.
Selected because the BigEarthNet.txt dataset paper's own baseline uses this exact
model, and its 1B size fits an 8GB consumer GPU with LoRA (full 7B-class alternatives
like GeoChat did not).

## Dataset
**BigEarthNet.txt** (BIFOLD-BigEarthNetv2-0, CDLA-Permissive-1.0 license):
~9.5M image-text annotations over co-registered Sentinel-1 SAR and Sentinel-2
multispectral patches. Used a stratified sample across categories
(presence, area, count, adjacency, season, climate zone, country) plus captioning:
~8,000 train rows, 960 validation rows. Evaluated on the dataset's official
manually-verified "bench" split (15,029 rows), using only RGB (B04/B03/B02) bands.

## Adaptation method
LoRA (Low-Rank Adaptation) via HuggingFace PEFT.
- Target modules: q_proj, k_proj, v_proj, o_proj (language model attention layers,
  verified via direct model introspection, not assumed)
- r=8, alpha=16, dropout=0.05
- Trainable parameters: 2,260,992 / 940,454,016 total (0.24%)
- Precision: bfloat16, with gradient checkpointing (required to fit 8GB VRAM)
- 1 epoch, ~7,000 steps, batch size 1 with 8-step gradient accumulation

## Results
Evaluated on 87 binary/mcq questions from the bench split (exact-match scoring
after extracting the letter/yes-no from generated text):

| Model | Accuracy |
|---|---|
| Base (unadapted) | 14.94% |
| LoRA-adapted | 43.68% |
| **Improvement** | **+28.74 points** |

## Known limitations (honest disclosure)
- The base model largely fails not from lack of visual understanding but because
  it answers in full sentences rather than the expected short format
  (letter/yes-no) — part of the improvement reflects learning response format,
  not only visual understanding.
- The adapted model shows some answer-frequency bias (e.g. over-predicting 'a'
  for mcq, 'no' for binary questions) — a known LoRA overfitting pattern with a
  single epoch on this size of sample. A second run with a lower learning rate
  and validation-based early stopping would likely reduce this.
- Training data subset was limited by available extracted image patches
  (not the full BigEarthNet-v2.0 archive), so some sampled rows lacked images
  and were skipped.

## Inference interface
See `infer.py`:
```python
from modules.model_adaptation.infer import RemoteSensingVLM
vlm = RemoteSensingVLM()
result = vlm.infer(image="path/to/image.tif_or_PIL", query="Describe this image.")
```
Returns dict matching team I/O contract: success, answer, confidence,
visual_output, model, error.
