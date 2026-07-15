"""Image decoding and encoding helpers for the HTTP boundary."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Composition is scale-invariant, so analysis runs on a bounded copy: the
# perception models downsample internally anyway and the summed-area tables
# are O(pixels) in memory.
MAX_ANALYSIS_SIDE = 1024
SALIENCY_PREVIEW_WIDTH = 256


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB array, honoring EXIF orientation.

    Phone cameras store rotation as metadata; without exif_transpose a
    portrait shot would be analyzed lying on its side.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise ValueError("uploaded file is not a decodable image") from exc
    transposed = ImageOps.exif_transpose(image)
    return np.asarray(transposed.convert("RGB"))


def downscale(image: np.ndarray, max_side: int = MAX_ANALYSIS_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def saliency_data_uri(saliency: np.ndarray, width: int = SALIENCY_PREVIEW_WIDTH) -> str:
    """Encode the saliency map as a small grayscale PNG data URI."""
    height = max(1, round(saliency.shape[0] * width / saliency.shape[1]))
    preview = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_AREA)
    gray = (np.clip(preview, 0.0, 1.0) * 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("failed to encode saliency preview")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")
