"""
modules/multimodal/fusion.py

Combines the separate optical and SAR feature vectors (from
feature_extraction.py) into a single joint representation for the task
head to consume.

Two fusion strategies are provided:
- concat_fusion: simplest, most robust option — concatenate both vectors
  and project down with a small MLP. Good default, easy to train, hard
  to get wrong.
- gated_fusion: lets the model learn to weight each modality's
  contribution per-input (useful if one modality is sometimes
  uninformative, e.g. SAR over featureless water vs optical over cloud
  cover). More expressive, more parameters to train, more prone to
  overfitting on a small dataset.

Start with concat_fusion; only switch to gated_fusion if you have enough
training data to justify the extra parameters and see concat underfitting.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Concatenates optical and SAR feature vectors, then projects down to
    a fixed-size joint representation via a small MLP.
    """

    def __init__(self, optical_dim: int = 512, sar_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(optical_dim + sar_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, optical_vec: torch.Tensor, sar_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([optical_vec, sar_vec], dim=-1)
        return self.proj(combined)


class GatedFusion(nn.Module):
    """
    Learns a per-dimension gate (0-1) controlling how much each modality
    contributes to the fused output. Requires optical_dim == sar_dim.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.output_dim = feature_dim

    def forward(self, optical_vec: torch.Tensor, sar_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([optical_vec, sar_vec], dim=-1)
        gate_weights = self.gate(combined)  # (batch, feature_dim), values in [0, 1]
        return gate_weights * optical_vec + (1 - gate_weights) * sar_vec


class MultimodalFusion:
    """
    Thin wrapper exposing a simple array-in, array-out interface over
    the chosen fusion strategy, matching the style of
    MultimodalFeatureExtractor in feature_extraction.py.
    """

    def __init__(
        self,
        strategy: str = "concat",
        feature_dim: int = 512,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if strategy == "concat":
            self.model = ConcatFusion(feature_dim, feature_dim, feature_dim)
        elif strategy == "gated":
            self.model = GatedFusion(feature_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy!r}. Use 'concat' or 'gated'.")

        self.model.to(self.device).eval()
        self.output_dim = self.model.output_dim

    def fuse(self, optical_vector: np.ndarray, sar_vector: np.ndarray) -> np.ndarray:
        """
        Fuse a single optical + SAR feature vector pair.

        Args:
            optical_vector: (feature_dim,) numpy array from
                MultimodalFeatureExtractor.
            sar_vector: (feature_dim,) numpy array from
                MultimodalFeatureExtractor.

        Returns:
            (output_dim,) numpy array — the joint representation.
        """
        opt_t = torch.from_numpy(optical_vector).float().unsqueeze(0).to(self.device)
        sar_t = torch.from_numpy(sar_vector).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            fused = self.model(opt_t, sar_t)

        return fused.squeeze(0).cpu().numpy()
