"""Fast rectangular queries over a saliency map via summed-area tables.

Building the tables is O(pixels) once; each region query is O(1) after that,
which is what makes scoring hundreds of candidate crops cheap. Alongside the
plain mass table we keep x- and y-weighted moment tables so the weighted
centroid of any rectangle is also O(1).
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-9


def _summed_area_table(values: np.ndarray) -> np.ndarray:
    table = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return table


class SaliencyStats:
    def __init__(self, saliency: np.ndarray):
        if saliency.ndim != 2:
            raise ValueError(f"saliency must be 2D, got shape {saliency.shape}")
        self.height, self.width = saliency.shape
        # Pixel centers sit at index + 0.5.
        xs = np.arange(self.width, dtype=np.float64) + 0.5
        ys = np.arange(self.height, dtype=np.float64) + 0.5
        self._mass = _summed_area_table(saliency)
        self._moment_x = _summed_area_table(saliency * xs[np.newaxis, :])
        self._moment_y = _summed_area_table(saliency * ys[:, np.newaxis])

    @property
    def total(self) -> float:
        return float(self._mass[-1, -1])

    def region_sum(self, x: int, y: int, w: int, h: int) -> float:
        """Sum of saliency inside the rectangle, clipped to image bounds."""
        return self._rect(self._mass, x, y, w, h)

    def region_centroid(self, x: int, y: int, w: int, h: int) -> tuple[float, float] | None:
        """Weighted center of mass (pixel coords) inside the rectangle, or
        None when the region holds no mass."""
        mass = self.region_sum(x, y, w, h)
        if mass <= _EPSILON:
            return None
        cx = self._rect(self._moment_x, x, y, w, h) / mass
        cy = self._rect(self._moment_y, x, y, w, h) / mass
        return cx, cy

    def _rect(self, table: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        x1 = max(0, min(x, self.width))
        y1 = max(0, min(y, self.height))
        x2 = max(x1, min(x + w, self.width))
        y2 = max(y1, min(y + h, self.height))
        return float(table[y2, x2] - table[y1, x2] - table[y2, x1] + table[y1, x1])
