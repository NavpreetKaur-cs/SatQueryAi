## Results (updated — evaluated on 2,000 bench questions, not 87)

Initial evaluation used only 87 questions, which gave an unreliable base
accuracy (14.94%). Re-running on a proper 2,000-question random sample
from the bench split gives a more trustworthy picture:

| Model | Accuracy (binary/mcq, 2000 questions) |
|---|---|
| Base (unadapted) | 24.41% |
| Adapted (LoRA, final) | 42.34% |
| **Improvement** | **+17.93 points** |

### Per-category breakdown (adapted model)
- Strongest: adjacency (48.86%), count (46.13%), area (43.15%)
- Weakest: relative position (24.68%), country (25.00%) — near chance,
  a genuine limitation worth flagging
- climate zone (30.30%) and season (35.19%) also underperform the average

### Known limitations
- The model shows answer-frequency bias: it disproportionately answers
  "no" and "b" rather than reasoning per-image. A second training run
  with a balanced dataset, lower learning rate, and early stopping
  (see train_lora_v2.py) produced a near-identical result (42.46%,
  +0.12 over the original), indicating the bias is not fully explained
  by data imbalance or learning rate alone.
- Per-category accuracy varies substantially — aggregate accuracy hides
  weak performance on relative-position and country-identification questions.
- Single epoch (V1) / up to 3 epochs (V2) on an 8,000-10,240 row sample —
  not the full 9.5M-row dataset.
