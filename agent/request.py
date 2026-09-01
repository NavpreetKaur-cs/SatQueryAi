from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageInput:
    """
    Represents a single input image.

    Attributes:
        path: Path to the image.
        modality: optical, sar, or None.
        acquisition_time: Optional acquisition date/time.
        format: Optional image format such as GeoTIFF or TIFF.
    """

    path: str
    modality: Optional[str] = None
    acquisition_time: Optional[str] = None
    format: Optional[str] = None


@dataclass
class AgentRequest:
    """
    Input received by SatQuery AI agent.
    """

    query: str

    images: List[ImageInput] = field(
        default_factory=list
    )

    metadata: Optional[Dict[str, Any]] = None