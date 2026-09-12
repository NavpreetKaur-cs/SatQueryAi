"""
CDVQA Model Interface

This module provides the model interface for change-based Visual Question Answering.

Currently, the change_analysis module uses an OpenCV baseline for local development.
This file is reserved for a trained CDVQA model for improved accuracy.

Model Options:
- Train on CDVQA dataset using a vision-language model (e.g., CLIP, LLaVA)
- Fine-tune on bi-temporal satellite imagery for change detection
- Use semantic segmentation labels from SECOND dataset for land-cover mapping

Expected Interface:
    predict(before_image_path, after_image_path, query) -> Dict[str, Any]
        Returns: {
            "answer": str,
            "confidence": float,
            "reasoning": str,
            "class_changes": Dict[str, float]
        }
"""

# Placeholder for trained CDVQA model
# To implement: Load a trained model checkpoint and provide inference interface
