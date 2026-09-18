"""
Handles everything about getting an image onto disk and back out again:
saving uploads, decoding inline dataUrls (what the current frontend sends),
and generating a browser-viewable PNG preview for formats browsers can't
render directly (TIFF/GeoTIFF).

STORAGE NOTE: image metadata is kept in an in-memory dict (_REGISTRY) that
resets whenever the server restarts. That's fine for a hackathon demo but
not for anything long-lived — if this needs to survive restarts, swap
_REGISTRY for a small SQLite table or a JSON sidecar file. Flagging this
now rather than silently shipping something that looks durable but isn't.

GEOTIFF NOTE: real multi-band GeoTIFF (with CRS/geo metadata) needs
`rasterio`, which is NOT in requirements.txt by default because it pulls in
GDAL and is a heavy/finicky install. This module tries Pillow first (which
handles PNG/JPEG and simple single-band TIFF), and only reaches for
rasterio if it's installed. If you need real GeoTIFF support, run:
    pip install rasterio
and add it to requirements.txt (tell the team first, per rule 8).
"""

import base64
import io
import re
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

# repo_root/uploads, regardless of where the server process is launched from
REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "uploads"
ORIGINALS_DIR = UPLOAD_DIR / "originals"
PREVIEWS_DIR = UPLOAD_DIR / "previews"
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# imageId -> { path, previewPath, width, height, format }
_REGISTRY: dict[str, dict] = {}

_DATA_URL_RE = re.compile(r"^data:image/(?P<ext>\w+);base64,(?P<data>.+)$", re.DOTALL)


class ImageStoreError(Exception):
    """Raised for any problem reading/saving an image — caught by main.py
    and turned into a contract-shaped { success: false, error } response."""


def _needs_conversion(fmt: str) -> bool:
    return fmt.upper() in ("TIFF",)


def _build_preview(image: Image.Image, image_id: str) -> str:
    """Save an 8-bit RGB PNG preview for formats the browser can't render
    natively, and return its web-servable path (relative to /uploads)."""
    preview = image.convert("RGB")
    preview_path = PREVIEWS_DIR / f"{image_id}.png"
    preview.save(preview_path, format="PNG")
    return f"/uploads/previews/{image_id}.png"


def _register(image_id: str, image: Image.Image, original_path: Path, fmt: str) -> dict:
    width, height = image.size
    if _needs_conversion(fmt):
        preview_url = _build_preview(image, image_id)
    else:
        preview_url = f"/uploads/originals/{original_path.name}"

    record = {
        "imageId": image_id,
        "path": str(original_path),
        "previewUrl": preview_url,
        "width": width,
        "height": height,
        "format": fmt,
    }
    _REGISTRY[image_id] = record
    return record


def _open_with_fallback(path: Path) -> Image.Image:
    """Try Pillow first; fall back to rasterio (if installed) for
    multi-band GeoTIFFs Pillow can't read correctly."""
    try:
        img = Image.open(path)
        img.load()  # force-read now so truncated/corrupt files fail here, not later
        return img
    except Exception as pillow_error:
        try:
            import numpy as np
            import rasterio
        except ImportError:
            raise ImageStoreError(
                f"Could not read '{path.name}' with Pillow ({pillow_error}). "
                "If this is a multi-band GeoTIFF, install rasterio "
                "(`pip install rasterio`, and tell the team per rule 8) "
                "for proper GeoTIFF support."
            ) from pillow_error

        with rasterio.open(path) as src:
            band_count = min(src.count, 3)
            arr = src.read(list(range(1, band_count + 1)))  # (bands, H, W)
            arr = np.moveaxis(arr, 0, -1)  # -> (H, W, bands)
            if band_count == 1:
                arr = arr.repeat(3, axis=2)
            # simple min/max stretch to 8-bit for a viewable preview —
            # NOT radiometrically calibrated, just enough to display
            arr = arr.astype("float32")
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255
            arr = arr.astype("uint8")
            return Image.fromarray(arr, mode="RGB")


def save_upload_bytes(data: bytes, filename: str) -> dict:
    image_id = uuid.uuid4().hex
    suffix = Path(filename).suffix or ".png"
    original_path = ORIGINALS_DIR / f"{image_id}{suffix}"
    original_path.write_bytes(data)

    image = _open_with_fallback(original_path)
    fmt = (image.format or suffix.lstrip(".").upper() or "UNKNOWN")
    return _register(image_id, image, original_path, fmt)


def save_data_url(data_url: str) -> dict:
    match = _DATA_URL_RE.match(data_url)
    if not match:
        raise ImageStoreError("Malformed data URL — expected 'data:image/<ext>;base64,...'.")
    ext = match.group("ext")
    raw = base64.b64decode(match.group("data"))

    image_id = uuid.uuid4().hex
    original_path = ORIGINALS_DIR / f"{image_id}.{ext}"
    original_path.write_bytes(raw)

    image = _open_with_fallback(original_path)
    fmt = image.format or ext.upper()
    return _register(image_id, image, original_path, fmt)


def resolve_image(image_id: Optional[str], data_url: Optional[str]) -> dict:
    """Given an ImageInput's fields, return its stored record — looking it
    up if imageId was given, or decoding+storing it fresh if a dataUrl was
    given instead. Exactly one of the two is expected."""
    if image_id:
        record = _REGISTRY.get(image_id)
        if not record:
            raise ImageStoreError(f"No uploaded image found with imageId '{image_id}'.")
        return record
    if data_url:
        return save_data_url(data_url)
    raise ImageStoreError("Image input must include either imageId or dataUrl.")
