\# Optical + SAR Multimodal Analysis — Person 4



\## Overview

Handles paired optical/multispectral and SAR satellite imagery, validates

that a given pair is usable together, extracts information from each

modality, fuses them into a joint representation, and answers land-cover

queries (built-up areas, water bodies, vegetation).



\## Architecture



```

interface.py          entry point matching agent/tools.py's multimodal\_tool contract

preprocessing.py       loads GeoTIFF/TIFF via rasterio, validates co-registration

&#x20;                       (CRS, geographic bounds, pixel dimensions), normalizes

&#x20;                       optical to \[0,1] and SAR to log/dB-scaled \[0,1]

feature\_extraction.py  two separate ResNet18 backbones (optical: 3-channel,

&#x20;                       SAR: 1-channel), output 512-dim feature vector each

fusion.py               combines the two 512-dim vectors (concat+MLP by

&#x20;                       default, gated fusion also available)

task\_head.py            keyword-routes the query to a category (built\_up /

&#x20;                       water / vegetation), then a small linear classifier

&#x20;                       predicts presence + confidence for that category

training/

&#x20; extract\_features\_cache.py   one-time pass over a labeled dataset,

&#x20;                               caches (fused\_vector, labels) to disk

&#x20; train\_task\_head.py           trains the classifier on the cache, with

&#x20;                               validation-based checkpoint selection

```



\## Data flow

```

optical.tif + sar.tif + query

&#x20;       │

&#x20;       ▼

&#x20; preprocessing.py    (load, validate, normalize)

&#x20;       │

&#x20;       ▼

feature\_extraction.py (ResNet18 x2 → two 512-dim vectors)

&#x20;       │

&#x20;       ▼

&#x20;   fusion.py          (→ one 512-dim joint vector)

&#x20;       │

&#x20;       ▼

&#x20;  task\_head.py        (query → category → presence + confidence)

&#x20;       │

&#x20;       ▼

{success, answer, confidence, model, visual\_output, metadata}

```



\## Current status



| Component | Status |

|---|---|

| Input validation | Done |

| Co-registration checking | Done |

| Normalization (optical + SAR) | Done |

| Feature extraction | Done (untrained/ImageNet-pretrained backbones — no internet access to download pretrained weights in this dev environment; see `feature\_extraction.py`'s `pretrained` flag) |

| Fusion | Done |

| Query → category routing | Done |

| Land-cover classifier | \*\*Untrained.\*\* Architecture and training scripts are complete and tested (see below), but no labeled `data/optical\_sar/` dataset has been sourced/prepared yet. Until trained weights are loaded, `interface.py` correctly reports "model not yet trained" rather than returning a meaningless confidence score. |

| Visual output (masks/overlays) | Not implemented. `visual\_output` is always `None` for now. |

| Agent integration | Done — `interface.py`'s `multimodal\_tool()` matches the exact contract in `agent/tools.py` and has been tested against it. |



\## How to train the classifier once labeled data is available



\*\*Handoff note:\*\* the pipeline below is complete and tested end-to-end

(see "Current status" above). The only missing piece is a labeled

`data/optical\_sar/` dataset. The steps below are exactly what's needed —

no other code changes required to go from "pipeline exists" to "model

trained."



\*\*Dataset dead-ends already tried (avoid repeating these):\*\*

\- `GFM-Bench/BigEarthNet` on HuggingFace — its custom loading script is

&#x20; broken (returns empty splits) as of testing on 2026-09-06, even with

&#x20; `trust\_remote\_code=True` and a pinned older `datasets` version.

\- Official BigEarthNet v2.0 on Zenodo (https://zenodo.org/records/10891137)

&#x20; — legitimate and complete, but 118GB total (54GB for SAR alone) as

&#x20; `.tar.zst` archives with no easy way to pull a small slice. Only

&#x20; realistic if you have a fast connection and can let it download for

&#x20; many hours, or access to a machine/mirror that already has it cached.

\- \*\*Worth trying next:\*\* YYX-OPT-SAR

&#x20; (https://github.com/yeyuanxin110/YYX-OPT-SAR, Google Drive hosted,

&#x20; \~150 co-registered optical+SAR pairs with 7-class pixel labels

&#x20; including water/houses/trees/low-vegetation) — much smaller, not yet

&#x20; attempted. Would need a script to convert its pixel-level label masks

&#x20; into the binary built\_up/water/vegetation presence labels this

&#x20; module's `index.csv` expects (see `CATEGORY\_KEYWORDS` in

&#x20; `task\_head.py` for the exact category names to match).



1\. Populate `data/optical\_sar/` with:

&#x20;  - `index.csv` with columns: `optical\_path, sar\_path, built\_up, water, vegetation`

&#x20;    (paths relative to `data/optical\_sar/`, labels as 0/1 presence flags)

&#x20;  - the referenced image files

2\. Extract and cache features:

&#x20;  ```

&#x20;  python -m modules.multimodal.training.extract\_features\_cache

&#x20;  ```

3\. Train the classifier:

&#x20;  ```

&#x20;  python -m modules.multimodal.training.train\_task\_head

&#x20;  ```

4\. In `interface.py`, load the trained weights when building the task head

&#x20;  (inside `\_get\_pipeline\_components()`):

&#x20;  ```python

&#x20;  task\_head = TaskHead(feature\_dim=512)

&#x20;  task\_head.load\_weights("models/multimodal/checkpoints/land\_cover\_head.pt")

&#x20;  ```



Suggested datasets: SEN12MS or WHU-OPT-SAR (paired optical + SAR with

land-cover labels).



\## Known limitations (honest disclosure)

\- The classifier has never been trained on real data — see above.

\- Feature extraction backbones are not ImageNet-pretrained in this dev

&#x20; environment (no internet access to torchvision's model hub); set

&#x20; `pretrained=True` in `feature\_extraction.py` once running somewhere

&#x20; with access, for a better starting point than random initialization.

\- Query routing is keyword-based, not a learned language model — queries

&#x20; that don't mention "built-up/water/vegetation" (or close synonyms)

&#x20; won't be recognized. This is a reasonable v0 given the task doc's

&#x20; named example categories, but won't generalize to open-ended VQA.

\- No visual output (overlay/mask) is generated yet.



\## Testing

All pipeline stages have manual test scripts (`\_manual\_test\_\*.py`,

kept local/not committed) covering: valid pairs, missing images, bad

extensions, mismatched co-registration, degenerate inputs, and the full

wired pipeline end to end via `interface.py`.

