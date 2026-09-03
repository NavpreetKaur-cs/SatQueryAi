"""
modules/multimodal/preprocessing.py

Loads and validates a paired optical + SAR image, and normalizes each
modality so downstream feature extraction gets consistent, well-scaled
input.

This module is intentionally independent of the agent/tools.py contract —
interface.py is the only file that talks to the agent. This file just
does image I/O and math, so it's easy to test on its own with any two
GeoTIFF/TIFF files.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.coords import BoundingBox


@dataclass
class LoadedImage:
    """A single loaded raster, with the metadata needed for validation."""
    array: np.ndarray          # shape: (bands, height, width)
    crs: Optional[str]
    transform: rasterio.Affine
    bounds: BoundingBox
    width: int
    height: int
    path: str


class CoRegistrationError(ValueError):
    """Raised when the optical/SAR pair is not usable together."""
    pass


def load_image(path: str) -> LoadedImage:
    """
    Load a raster file (GeoTIFF/TIFF, or any rasterio-supported format)
    and return it with the metadata needed for co-registration checks.
    """
    with rasterio.open(path) as src:
        array = src.read()  # (bands, height, width)
        return LoadedImage(
            array=array,
            crs=src.crs.to_string() if src.crs else None,
            transform=src.transform,
            bounds=src.bounds,
            width=src.width,
            height=src.height,
            path=path,
        )


def check_co_registration(
    optical: LoadedImage,
    sar: LoadedImage,
    bounds_tolerance: float = 1e-3,
) -> None:
    """
    Verify the optical and SAR images cover the same geographic area at
    compatible resolution, so they can be fused pixel-for-pixel.

    Raises CoRegistrationError with a clear message if they don't match.
    If either image has no CRS (e.g. a plain PNG/JPEG test file with no
    geospatial metadata), the CRS/bounds checks are skipped and only the
    pixel-dimension check runs — this keeps the function usable during
    early development with non-georeferenced test images.
    """
    has_geo_metadata = optical.crs is not None and sar.crs is not None

    if has_geo_metadata:
        if optical.crs != sar.crs:
            raise CoRegistrationError(
                f"CRS mismatch: optical={optical.crs}, sar={sar.crs}. "
                "Reproject one image to match the other before fusion."
            )

        ob, sb = optical.bounds, sar.bounds
        if not (
            abs(ob.left - sb.left) <= bounds_tolerance
            and abs(ob.bottom - sb.bottom) <= bounds_tolerance
            and abs(ob.right - sb.right) <= bounds_tolerance
            and abs(ob.top - sb.top) <= bounds_tolerance
        ):
            raise CoRegistrationError(
                f"Bounding boxes do not match within tolerance "
                f"({bounds_tolerance}): optical={ob}, sar={sb}. "
                "Images must be co-registered (same geographic extent)."
            )

    if (optical.width, optical.height) != (sar.width, sar.height):
        raise CoRegistrationError(
            f"Pixel dimension mismatch: optical={optical.width}x{optical.height}, "
            f"sar={sar.width}x{sar.height}. Resample one image to match "
            "the other's resolution before fusion."
        )


def normalize_optical(array: np.ndarray) -> np.ndarray:
    """
    Normalize an optical/multispectral array to [0, 1] float32.

    Handles both 8-bit (0-255) and reflectance-style (0-1 or larger
    integer range) inputs by scaling based on the observed max value.
    """
    arr = array.astype(np.float32)
    max_val = arr.max()

    if max_val <= 1.0:
        # Already reflectance-scaled
        return arr
    elif max_val <= 255.0:
        return arr / 255.0
    else:
        # Likely 12/16-bit sensor data; scale by observed max
        return arr / max_val


def normalize_sar(array: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Normalize a SAR array for model input.

    SAR backscatter values span a huge dynamic range (often several
    orders of magnitude), so raw linear values are not directly usable.
    This converts linear intensity to a log (dB-like) scale, then
    min-max normalizes to [0, 1]. If the array already looks log-scaled
    (contains negative values, typical of dB-scaled products), it skips
    the log step and only min-max normalizes.
    """
    arr = array.astype(np.float32)

    looks_already_log_scaled = arr.min() < 0
    if not looks_already_log_scaled:
        arr = 10.0 * np.log10(np.clip(arr, epsilon, None))

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < epsilon:
        # Flat/degenerate image (e.g. a blank test file) — avoid divide by zero
        return np.zeros_like(arr)

    return (arr - arr_min) / (arr_max - arr_min)

def load_and_validate_pair(
    optical_path: str,
    sar_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load, validate, and normalize an optical + SAR image pair.

    Returns:
        (optical_array, sar_array): both float32 arrays, shape
        (bands, height, width), normalized and ready for feature
        extraction.

    Raises:
        CoRegistrationError: if the pair isn't usable together.
        rasterio.errors.RasterioIOError: if a file can't be read.
    """
    optical = load_image(optical_path)
    sar = load_image(sar_path)

    check_co_registration(optical, sar)

    optical_norm = normalize_optical(optical.array)
    sar_norm = normalize_sar(sar.array)

    return optical_norm, sar_norm