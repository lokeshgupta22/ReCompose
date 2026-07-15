"""Fast rectangular queries over a saliency map via a summed-area table.

Building the table is O(pixels) once; each region query is O(1) after that,
which is what makes scoring hundreds of candidate crops cheap.
"""

from __future__ import annotations

import numpy as np


class SaliencyStats:
    def __init__(self, saliency: np.ndarray):
        if saliency.ndim != 2:
            raise ValueError(f"saliency must be 2D, got shape {saliency.shape}")
        self.height, self.width = saliency.shape
        self._integral = np.zeros((self.height + 1, self.width + 1), dtype=np.float64)
        self._integral[1:, 1:] = saliency.cumsum(axis=0).cumsum(axis=1)

    @property
    def total(self) -> float:
        return float(self._integral[-1, -1])

    def region_sum(self, x: int, y: int, w: int, h: int) -> float:
        """Sum of saliency inside the rectangle, clipped to image bounds."""
        x1 = max(0, min(x, self.width))
        y1 = max(0, min(y, self.height))
        x2 = max(x1, min(x + w, self.width))
        y2 = max(y1, min(y + h, self.height))
        ii = self._integral
        return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])
