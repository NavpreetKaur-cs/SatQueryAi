# Optical + SAR Multimodal Analysis — Rohan

## Overview
Handles paired optical/multispectral and SAR satellite imagery, validates
that a given pair is usable together, extracts information from each
modality, fuses them into a joint representation, and answers land-cover
queries (built-up areas, water bodies, vegetation).

## Architecture

```
interface.py          entry point matching agent/tools.py's multimodal_tool contract
preprocessing.py       loads GeoTIFF/TIFF via rasterio, validates co-registration
                        (CRS, geographic bounds, pixel dimensions), normalizes
                        optical to [0,1] and SAR to log/dB-scaled [0,1]
feature_extraction.py  two separate ResNet18 backbones (optical: 3-channel,
                        SAR: 1-channel), output 512-dim feature vector each
fusion.py               combines the two 512-dim vectors (concat+MLP by
                        default, gated fusion also available)
task_head.py            keyword-routes the query to a category (built_up /
                        water / vegetation), then a small linear classifier
                        predicts presence + confidence for that category
training/
  extract_features_cache.py   one-time pass over a labeled dataset,
                                caches (fused_vector, labels) to disk
  train_task_head.py           trains the classifier on the cache, with
                                validation-based checkpoint selection
```

## Data flow
```
optical.tif + sar.tif + query
        │
        ▼
  preprocessing.py    (load, validate, normalize)
        │
        ▼
feature_extraction.py (ResNet18 x2 → two 512-dim vectors)
        │
        ▼
    fusion.py          (→ one 512-dim joint vector)
        │
        ▼
   task_head.py        (query → category → presence + confidence)
        │
        ▼
{success, answer, confidence, model, visual_output, metadata}
```

## Current status

| Component | Status |
|---|---|
| Input validation | Done |
| Co-registration checking | Done |
| Normalization (optical + SAR) | Done |
| Feature extraction | Done (untrained/ImageNet-pretrained backbones — no internet access to download pretrained weights in this dev environment; see `feature_extraction.py`'s `pretrained` flag) |
| Fusion | Done |
| Query → category routing | Done |
| Land-cover classifier | **Untrained.** Architecture and training scripts are complete and tested (see below), but no labeled `data/optical_sar/` dataset has been sourced/prepared yet. Until trained weights are loaded, `interface.py` correctly reports "model not yet trained" rather than returning a meaningless confidence score. |
| Visual output (masks/overlays) | Done — generates an overlay image (colored border per category + text banner with prediction/confidence) via `visualize.py`. This is a whole-image classification overlay, not a per-pixel segmentation mask; upgrading to true pixel-level masks would need a segmentation model instead of/alongside the current classifier. |
| Agent integration | Done — `interface.py`'s `multimodal_tool()` matches the exact contract in `agent/tools.py` and has been tested against it. |

## How to train the classifier once labeled data is available

**Handoff note:** the pipeline below is complete and tested end-to-end
(see "Current status" above). The only missing piece is a labeled
`data/optical_sar/` dataset. The steps below are exactly what's needed —
no other code changes required to go from "pipeline exists" to "model
trained."

**Dataset dead-ends already tried (avoid repeating these):**
- `GFM-Bench/BigEarthNet` on HuggingFace — its custom loading script is
  broken (returns empty splits) as of testing on 2026-09-06, even with
  `trust_remote_code=True` and a pinned older `datasets` version.
- Official BigEarthNet v2.0 on Zenodo (https://zenodo.org/records/10891137)
  — legitimate and complete, but 118GB total (54GB for SAR alone) as
  `.tar.zst` archives with no easy way to pull a small slice. Only
  realistic if you have a fast connection and can let it download for
  many hours, or access to a machine/mirror that already has it cached.
- **Worth trying next:** YYX-OPT-SAR
  (https://github.com/yeyuanxin110/YYX-OPT-SAR, Google Drive hosted,
  ~150 co-registered optical+SAR pairs with 7-class pixel labels
  including water/houses/trees/low-vegetation) — much smaller, not yet
  attempted. Would need a script to convert its pixel-level label masks
  into the binary built_up/water/vegetation presence labels this
  module's `index.csv` expects (see `CATEGORY_KEYWORDS` in
  `task_head.py` for the exact category names to match).

1. Populate `data/optical_sar/` with:
   - `index.csv` with columns: `optical_path, sar_path, built_up, water, vegetation`
     (paths relative to `data/optical_sar/`, labels as 0/1 presence flags)
   - the referenced image files
2. Extract and cache features:
   ```
   python -m modules.multimodal.training.extract_features_cache
   ```
3. Train the classifier:
   ```
   python -m modules.multimodal.training.train_task_head
   ```
4. In `interface.py`, load the trained weights when building the task head
   (inside `_get_pipeline_components()`):
   ```python
   task_head = TaskHead(feature_dim=512)
   task_head.load_weights("models/multimodal/checkpoints/land_cover_head.pt")
   ```

Suggested datasets: SEN12MS or WHU-OPT-SAR (paired optical + SAR with
land-cover labels).

## Known limitations (honest disclosure)
- The classifier has never been trained on real data — see above.
- Feature extraction backbones are not ImageNet-pretrained in this dev
  environment (no internet access to torchvision's model hub); set
  `pretrained=True` in `feature_extraction.py` once running somewhere
  with access, for a better starting point than random initialization.
- Query routing is keyword-based, not a learned language model — queries
  that don't mention "built-up/water/vegetation" (or close synonyms)
  won't be recognized. This is a reasonable v0 given the task doc's
  named example categories, but won't generalize to open-ended VQA.
- Visual output is a whole-image overlay (colored border + text banner), not a per-pixel segmentation mask.
## Testing
All pipeline stages have manual test scripts (`_manual_test_*.py`,
kept local/not committed) covering: valid pairs, missing images, bad
extensions, mismatched co-registration, degenerate inputs, and the full
wired pipeline end to end via `interface.py`.