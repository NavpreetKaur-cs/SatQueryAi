"""
INTEGRATION POINT for Person 5's agent/controller.

CONFIRMED against the real agent/controller.py, agent/request.py, and
agent/response.py (synced from the repo) — the actual interface is:

    from agent.controller import process
    from agent.request import AgentRequest, ImageInput as AgentImageInput

    result = process(AgentRequest(query=..., images=[AgentImageInput(...), ...], metadata=...))
    result.to_dict()  # -> {success, answer, task, confidence, model, visual_output, trace, error, metadata}

This replaces an earlier guessed `handle_query(query, images, metadata)`
signature that never existed in the real module — that guess would have
silently kept using the mock forever (the ImportError was caught with no
visible warning). Verified by actually running agent.controller.process()
against the real request/response dataclasses before writing this.

Key differences from the earlier guess, worth knowing:
  - Images are a LIST (List[ImageInput]), not a {before, after} dict.
    Each ImageInput has: path, modality ('optical'/'sar'/None),
    acquisition_time, format — no width/height.
  - The agent auto-classifies the task ("single_image" / "change_analysis"
    / "multimodal") from the query + image count via its own router —
    this is exactly why the frontend has no query-type selector.
  - AgentResult already includes a full execution summary at
    metadata['execution_summary'], matching the Person 6 role doc's spec
    (selected task, tools/models used, parameters, outputs) almost exactly.

STILL UNCONFIRMED: the real shape of `visual_output` coming out of
modules/single_image, modules/change_analysis, modules/multimodal
(Person 1-4's code) — that's determined by agent/executor.py calling into
their modules, which hasn't been tested here. _normalize_visual_output()
below is defensive rather than a guarantee: it passes through anything
that already looks like our region-list shape, and doesn't crash on
anything else, but the exact shape should be confirmed with a real
end-to-end run once those modules are wired into the agent's executor.
"""

import random
import re
import warnings
from typing import Optional

try:
    from agent.controller import process as _agent_process
    from agent.request import AgentRequest, ImageInput as AgentImageInput
    USING_REAL_AGENT = True
except ImportError as e:
    _agent_process = None
    AgentRequest = None
    AgentImageInput = None
    USING_REAL_AGENT = False
    # This except fires both when agent/ genuinely doesn't exist yet AND when
    # it exists but a dependency it needs (joblib, scikit-learn, torch, ...)
    # isn't installed — those look identical to Python (both raise
    # ImportError/ModuleNotFoundError) but mean very different things. Print
    # the real reason so a missing-dependency case doesn't get silently
    # mistaken for "the agent isn't ready yet".
    warnings.warn(
        f"Falling back to the mock agent — could not import agent.controller. "
        f"If agent/ already exists in your repo, this is very likely a missing "
        f"dependency (e.g. joblib, scikit-learn — the router's task classifier "
        f"needs these) rather than the agent not existing. "
        f"Underlying error: {type(e).__name__}: {e}"
    )


def _normalize_visual_output(visual_output) -> list:
    if not visual_output:
        return []
    if isinstance(visual_output, list):
        return visual_output
    if isinstance(visual_output, dict) and "regions" in visual_output:
        return visual_output["regions"] or []
    # File paths and other non-region visuals are exposed separately.
    return []


def _run_real_agent(query: str, images: dict, metadata: dict) -> dict:
    before = images["before"]
    after = images.get("after")

    agent_images = [
        AgentImageInput(
            path=before["path"],
            modality=before.get("modality") or metadata.get("beforeModality"),
        )
    ]
    if after:
        agent_images.append(
            AgentImageInput(
                path=after["path"],
                modality=after.get("modality") or metadata.get("afterModality"),
            )
        )

    request = AgentRequest(query=query, images=agent_images, metadata=metadata)
    result = _agent_process(request)
    d = result.to_dict()

    return {
        "success": d.get("success", False),
        "result": d.get("answer", ""),
        "confidence": d.get("confidence"),
        "regions": _normalize_visual_output(d.get("visual_output")),
        "queryType": d.get("task"),
        "modelsUsed": [d.get("model")] if d.get("model") else [],
        "executionSummary": d.get("metadata", {}).get("execution_summary"),
        "visualOutput": d.get("visual_output"),
        "error": d.get("error"),
    }


# ---------------------------------------------------------------------------
# Mock fallback — used only when agent/controller.py isn't importable yet
# (e.g. running the backend standalone before pulling the agent's code).
# ---------------------------------------------------------------------------

def _clamp_box(x, y, w, h, img_w, img_h):
    cx = max(0, min(x, img_w - 4))
    cy = max(0, min(y, img_h - 4))
    cw = max(4, min(w, img_w - cx))
    ch = max(4, min(h, img_h - cy))
    return [cx, cy, cw, ch]


def _seeded_boxes(seed: int, count: int, img_w: int, img_h: int):
    rng = random.Random(seed)
    boxes = []
    for i in range(count):
        w = img_w * (0.1 + rng.random() * 0.12)
        h = img_h * (0.08 + rng.random() * 0.1)
        x = rng.random() * (img_w - w)
        y = rng.random() * (img_h - h)
        boxes.append({
            "id": f"r{seed}-{i}",
            "bbox": _clamp_box(x, y, w, h, img_w, img_h),
            "score": round(0.7 + rng.random() * 0.28, 2),
        })
    return boxes


_KEYWORD_RULES = [
    {
        "match": re.compile(r"waterlog|flood|inundat", re.I),
        "queryType": "compare",
        "answer": (
            "Approximately 14.6 hectares of cropland along the riverbank are newly "
            "waterlogged compared to the earlier pass — mostly the low-lying parcels "
            "on the southern bend."
        ),
        "confidence": 0.84,
    },
    {
        "match": re.compile(r"construction|new building|riverbank", re.I),
        "queryType": "compare",
        "answer": (
            "One new built-up cluster appears near the riverbank that was not "
            "present in the earlier image — roughly 3 structures across 0.4 hectares."
        ),
        "confidence": 0.79,
    },
    {
        "match": re.compile(r"landslide|shadow|scar", re.I),
        "queryType": "classify",
        "answer": (
            "This is most consistent with a landslide scar rather than a shadow — "
            "the boundary follows the slope contour and shows disturbed-soil "
            "backscatter rather than a hard illumination edge."
        ),
        "confidence": 0.71,
    },
    {
        "match": re.compile(r"how many|count", re.I),
        "queryType": "count",
        "answer": "Detected 4 distinct built-up structures in the selected region.",
        "confidence": 0.88,
    },
]

_FALLBACK = {
    "queryType": "locate",
    "answer": (
        "Here is the region most relevant to that query, based on the visual "
        "features detected in the current scene."
    ),
    "confidence": 0.62,
}


def _mock_handle_query(query: str, images: dict, metadata: dict) -> dict:
    before = images["before"]
    after = images.get("after")

    rule = next((r for r in _KEYWORD_RULES if r["match"].search(query)), _FALLBACK)

    regions = []
    if rule["queryType"] == "compare" and after:
        regions += [
            {**b, "label": "baseline region", "image": "before"}
            for b in _seeded_boxes(11, 2, before["width"], before["height"])
        ]
        regions += [
            {**b, "label": "changed region", "image": "after"}
            for b in _seeded_boxes(12, 3, after["width"], after["height"])
        ]
    else:
        regions = [
            {**b, "label": f"region {i + 1}"}
            for i, b in enumerate(_seeded_boxes(41, 3, before["width"], before["height"]))
        ]

    return {
        "success": True,
        "result": rule["answer"],
        "confidence": rule["confidence"],
        "regions": regions,
        "queryType": rule["queryType"],
        "modelsUsed": ["mock-agent-v0"],
        "executionSummary": None,
        "error": None,
    }


def run_query(query: str, images: dict, metadata: Optional[dict] = None) -> dict:
    metadata = metadata or {}
    if USING_REAL_AGENT:
        return _run_real_agent(query, images, metadata)
    return _mock_handle_query(query, images, metadata)
