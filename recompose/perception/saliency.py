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

U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_INPUT_SIZE = 320
# ImageNet channel statistics; U²-Net was trained with this normalization.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def default_model_dir() -> Path:
    return Path(os.environ.get("RECOMPOSE_CACHE_DIR", "~/.cache/recompose")).expanduser()


def _ensure_weights(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    urllib.request.urlretrieve(U2NET_URL, tmp_path)  # noqa: S310 - pinned https URL
    tmp_path.rename(model_path)
    return model_path


class SaliencyEstimator:
    """Predicts a HxW float32 map in [0, 1] of where the eye is drawn."""

    def __init__(self, model_path: Path | None = None):
        import onnxruntime as ort

        path = _ensure_weights(model_path or default_model_dir() / "u2net.onnx")
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
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
