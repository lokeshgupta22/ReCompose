"""Salient-object detection with U²-Net running directly on ONNX Runtime.

We deliberately avoid wrapper libraries (rembg et al.): they drag in alpha
matting dependencies we never use, and direct ONNX inference is the same
runtime path we ship in production.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Two variants of the same architecture, both pinned to the same release.
# u2netp ("portable") is the default: full u2net's first inference peaks
# over 512MB resident all by itself, which OOMs the Render free tier we
# deploy to. On zidane.jpg, 3 of 4 aspect ratios chose identical crops
# under both models - YOLO's hard constraints do the structural work,
# saliency only ranks - so the fidelity cost is modest.
MODEL_URLS = {
    "u2net": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    "u2netp": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
}
DEFAULT_SALIENCY_MODEL = "u2netp"
U2NET_INPUT_SIZE = 320
# ImageNet channel statistics; U²-Net was trained with this normalization.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def default_model_dir() -> Path:
    return Path(os.environ.get("RECOMPOSE_CACHE_DIR", "~/.cache/recompose")).expanduser()


def saliency_model_name() -> str:
    name = os.environ.get("RECOMPOSE_SALIENCY_MODEL", DEFAULT_SALIENCY_MODEL)
    if name not in MODEL_URLS:
        raise ValueError(
            f"unknown saliency model {name!r}; expected one of {sorted(MODEL_URLS)}"
        )
    return name


def _ensure_weights(model_path: Path, url: str) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 - pinned https URL
    tmp_path.rename(model_path)
    return model_path


class SaliencyEstimator:
    """Predicts a HxW float32 map in [0, 1] of where the eye is drawn."""

    def __init__(self, model_path: Path | None = None):
        from recompose.perception.onnx_session import create_session

        name = saliency_model_name()
        path = _ensure_weights(
            model_path or default_model_dir() / f"{name}.onnx", MODEL_URLS[name]
        )
        self._session = create_session(path)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: HxWx3 RGB uint8. Returns HxW float32 saliency in [0, 1]."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got shape {image.shape}")
        height, width = image.shape[:2]
        outputs = self._session.run(None, {self._input_name: self._preprocess(image)})
        # First output is the finest-resolution side output (d1 in the paper).
        prediction = outputs[0][0, 0]
        return self._postprocess(prediction, width, height)

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image, (U2NET_INPUT_SIZE, U2NET_INPUT_SIZE), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        peak = resized.max()
        if peak > 0:
            resized /= peak
        normalized = (resized - _MEAN) / _STD
        return normalized.transpose(2, 0, 1)[np.newaxis]

    @staticmethod
    def _postprocess(prediction: np.ndarray, width: int, height: int) -> np.ndarray:
        lo, hi = float(prediction.min()), float(prediction.max())
        if hi > lo:
            prediction = (prediction - lo) / (hi - lo)
        return cv2.resize(
            prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
