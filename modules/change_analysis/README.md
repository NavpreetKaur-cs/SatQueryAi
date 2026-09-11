# Change analysis

`change_analysis_tool(before, after, question, metadata=None)` is the agent-facing entry point. It accepts image paths (or `ImageInput` objects), detects changes, writes `change_map.png` and `change_overlay.png`, and answers general or cover-specific questions.

```python
from modules.change_analysis import change_analysis_tool

result = change_analysis_tool("before.png", "after.png", "Has vegetation changed?", {
    "output_dir": "outputs/example", "threshold": 24,
})
print(result["answer"], result["visual_output"])
```

This initial baseline uses OpenCV registration and RGB land-cover heuristics. It is suitable for local development and provides a stable interface for a later trained CDVQA model; it is not a substitute for validated geospatial classification.

Place CDVQA annotations and imagery under `data/cdvqa/`. The helpers in `cdvqa.py` load JSON/JSONL/CSV annotations with aliases such as `image1`/`image2`/`question`/`answer` and run a lightweight evaluation.
