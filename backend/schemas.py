"""
Request/response schemas for the SatQuery AI backend.

These mirror the team's shared I/O contract exactly:

    Input:  Images, Natural-language query, Metadata (if required)
    Output: success status, textual answer/result, confidence (if available),
            visual output path/data (if applicable), model/task metadata,
            error message (if execution fails)

The frontend (src/api/client.js) already speaks this exact shape — do not
rename fields here without telling the frontend team, per team rule 9.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """
    An image supplied to /query. Either reference a previously uploaded
    image by imageId (preferred — see POST /images), or pass a raw
    data URL directly (what the current frontend does today).
    """
    imageId: Optional[str] = None
    dataUrl: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    modality: Optional[str] = None


class ImagesInput(BaseModel):
    before: ImageInput
    after: Optional[ImageInput] = None


class QueryRequest(BaseModel):
    images: ImagesInput
    query: str
    metadata: Optional[dict] = Field(default_factory=dict)


class Region(BaseModel):
    id: str
    label: str
    bbox: List[float]  # [x, y, w, h] in the ORIGINAL image's pixel space, origin top-left
    score: Optional[float] = None
    image: Optional[str] = None  # 'before' | 'after' — only used in compare mode


class VisualOutput(BaseModel):
    regions: List[Region] = Field(default_factory=list)


class QueryResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    confidence: Optional[float] = None
    visual_output: Optional[VisualOutput] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    imageId: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    previewUrl: Optional[str] = None
    error: Optional[str] = None
