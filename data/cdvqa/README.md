# CDVQA dataset

The actual dataset files are not stored in GitHub because they are large.
Download the required dataset from its official source and place its imagery and
annotations in this folder.

For the change-analysis baseline, annotations may be JSON, JSONL, or CSV. Each
record needs a before image, after image, question, and answer. The loader
accepts these aliases:

| Canonical field | Accepted names |
| --- | --- |
| before image | `before_image`, `image1`, `img1`, `t1` |
| after image | `after_image`, `image2`, `img2`, `t2` |
| question | `question`, `query` |
| answer | `answer`, `label`, `gt_answer` |

```python
from modules.change_analysis.cdvqa import evaluate_cdvqa, load_cdvqa

records = load_cdvqa("data/cdvqa/annotations.json")
metrics = evaluate_cdvqa(records, image_root="data/cdvqa")
print(metrics["exact_substring_accuracy"])
```
