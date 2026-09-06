"""
modules/multimodal/feature_extraction.py

Turns normalized optical and SAR arrays (from preprocessing.py) into
fixed-length feature vectors, using separate CNN backbones per modality.

Two backbones because optical and SAR are physically very different
signals (optical: reflected light, usually 3+ bands; SAR: radar
backscatter, usually 1-2 bands, different noise characteristics/texture).
Sharing one backbone across both tends to hurt quality, so this keeps
them independent and lets fusion.py combine the results later.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

def _make_backbone(in_channels: int, pretrained: bool = False) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)

    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained and in_channels < 3:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight[:, :in_channels, :, :].mean(
                    dim=1, keepdim=True
                ).repeat(1, in_channels, 1, 1)
        model.conv1 = new_conv

    modules = list(model.children())[:-1]
    return nn.Sequential(*modules)


@dataclass
class ModalityFeatures:
    """Feature vector output for a single modality."""
    vector: np.ndarray  
    feature_dim: int


class MultimodalFeatureExtractor:
    """
    Wraps separate optical and SAR backbones and exposes a simple
    array-in, vector-out interface.

    Usage:
        extractor = MultimodalFeatureExtractor()
        opt_feat, sar_feat = extractor.extract(optical_array, sar_array)
    """

    def __init__(
        self,
        optical_channels: int = 3,
        sar_channels: int = 1,
        pretrained: bool = False,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.optical_backbone = _make_backbone(
            optical_channels, pretrained=pretrained
        ).to(self.device).eval()

        self.sar_backbone = _make_backbone(
            sar_channels, pretrained=pretrained
        ).to(self.device).eval()

        self.feature_dim = 512  # ResNet18's output dim before the fc layer

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert a (bands, height, width) numpy array into a
        (1, bands, height, width) float32 tensor on the target device.
        """
        tensor = torch.from_numpy(array).float().unsqueeze(0)
        return tensor.to(self.device)

    def _run_backbone(self, backbone: nn.Module, array: np.ndarray) -> ModalityFeatures:
        tensor = self._to_tensor(array)
        with torch.no_grad():
            out = backbone(tensor)          # (1, 512, 1, 1)
            vector = out.flatten(1).squeeze(0).cpu().numpy()  # (512,)
        return ModalityFeatures(vector=vector, feature_dim=vector.shape[0])

    def extract(
        self,
        optical_array: np.ndarray,
        sar_array: np.ndarray,
    ) -> "tuple[ModalityFeatures, ModalityFeatures]":
        """
        Extract feature vectors from a normalized optical + SAR pair.

        Args:
            optical_array: (bands, height, width) float array, normalized
                to [0, 1] by preprocessing.normalize_optical.
            sar_array: (bands, height, width) float array, normalized to
                [0, 1] by preprocessing.normalize_sar.

        Returns:
            (optical_features, sar_features), each a ModalityFeatures
            with a 512-dim vector.
        """
        optical_features = self._run_backbone(self.optical_backbone, optical_array)
        sar_features = self._run_backbone(self.sar_backbone, sar_array)
        return optical_features, sar_features