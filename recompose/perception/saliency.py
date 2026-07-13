"""Saliency estimation using U²-Net (via rembg's pretrained session)."""

from __future__ import annotations

import numpy as np
from PIL import Image


class SaliencyEstimator:
    """Predicts a per-pixel saliency map in [0, 1] using U²-Net.

    rembg downloads and caches the pretrained ONNX weights on first use
    (~170 MB, stored under ~/.u2net/).
    """

    def __init__(self, model_name: str = "u2net"):
        from rembg import new_session

        self._session = new_session(model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: HxWx3 RGB uint8. Returns HxW float32 saliency in [0, 1]."""
        from rembg import remove

        mask = remove(Image.fromarray(image), session=self._session, only_mask=True)
        return np.asarray(mask, dtype=np.float32) / 255.0
