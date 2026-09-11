"""
modules/multimodal/task_head.py

Turns a fused optical+SAR feature vector plus a natural-language query
into an answer.

Given the doc's example queries (built-up areas, vegetation, water
bodies, other land-cover), this starts as a small closed-set classifier:
keyword-match the query to a land-cover category, then run a trained
linear head over the fused vector to predict presence + confidence for
that category. This is much more tractable to get working and evaluate
than free-form VQA, and can be extended later (e.g. swapping in a real
VQA head or an LLM-based answer generator that consumes the fused vector
as context) without changing the interface this module exposes.

NOTE: the classifier head below is untrained (randomly initialized).
Its predictions are not meaningful yet — training data and a training
script still need to be built (likely once data/optical_sar/ has real
labeled examples). This file wires up the full inference path so the
agent integration works today, with an honest placeholder confidence
until real weights exist.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Keyword sets per land-cover category mentioned in the team's task doc.
CATEGORY_KEYWORDS: Dict[str, list] = {
    "built_up": ["built-up", "built up", "building", "urban", "city", "settlement"],
    "water": ["water", "river", "lake", "sea", "flood", "reservoir"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "green", "farmland"],
}

DEFAULT_CATEGORY = "unknown"


def detect_category(query: str) -> str:
    """
    Simple keyword-based routing from a free-text query to a land-cover
    category. Falls back to DEFAULT_CATEGORY if nothing matches, in
    which case the caller should treat the answer as low-confidence.
    """
    q = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(re.escape(kw), q) for kw in keywords):
            return category
    return DEFAULT_CATEGORY


class LandCoverHead(nn.Module):
    """
    Small linear classifier over the fused feature vector, predicting
    presence probability for each known land-cover category.
    """

    def __init__(self, input_dim: int = 512, categories: Optional[list] = None):
        super().__init__()
        self.categories = categories or list(CATEGORY_KEYWORDS.keys())
        self.linear = nn.Linear(input_dim, len(self.categories))

    def forward(self, fused_vector: torch.Tensor) -> torch.Tensor:
        logits = self.linear(fused_vector)
        return torch.sigmoid(logits)  # independent presence probability per category


@dataclass
class TaskAnswer:
    answer: str
    confidence: Optional[float]
    category: str


class TaskHead:
    """
    Wraps LandCoverHead and query-category detection into a simple
    fused_vector + query -> answer interface.
    """

    def __init__(self, feature_dim: int = 512, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head = LandCoverHead(feature_dim).to(self.device).eval()
        self.is_trained = False  # flip to True once real weights are loaded

    def load_weights(self, path: str) -> None:
        """Load trained weights once available. Not yet used."""
        state_dict = torch.load(path, map_location=self.device)
        self.head.load_state_dict(state_dict)
        self.head.eval()
        self.is_trained = True

    def answer_query(self, fused_vector: np.ndarray, query: str) -> TaskAnswer:
        category = detect_category(query)

        if category == DEFAULT_CATEGORY:
            return TaskAnswer(
                answer=(
                    "I couldn't map this query to a known category "
                    "(built-up area, water, or vegetation). "
                    "Try rephrasing with one of those terms."
                ),
                confidence=None,
                category=category,
            )

        vector_t = torch.from_numpy(fused_vector).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.head(vector_t).squeeze(0).cpu().numpy()

        category_index = self.head.categories.index(category)
        presence_prob = float(probs[category_index])

        if not self.is_trained:
            return TaskAnswer(
                answer=(
                    f"Model not yet trained — cannot reliably answer whether "
                    f"'{category.replace('_', ' ')}' is present."
                ),
                confidence=None,
                category=category,
            )

        present = presence_prob >= 0.5
        answer = (
            f"Yes, {category.replace('_', ' ')} appears to be present."
            if present
            else f"No, {category.replace('_', ' ')} does not appear to be present."
        )
        return TaskAnswer(answer=answer, confidence=presence_prob, category=category)
