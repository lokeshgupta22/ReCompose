"""Contract for weighted centroid queries on the summed-area tables."""

import numpy as np
import pytest

from recompose.saliency_stats import SaliencyStats


class TestRegionCentroid:
    def test_point_mass_centroid_is_that_pixel_center(self):
        saliency = np.zeros((50, 60), dtype=np.float32)
        saliency[40, 30] = 1.0
        cx, cy = SaliencyStats(saliency).region_centroid(0, 0, 60, 50)
        assert cx == pytest.approx(30.5)
        assert cy == pytest.approx(40.5)

    def test_uniform_mass_centroid_is_region_center(self):
        saliency = np.ones((100, 100), dtype=np.float32)
        cx, cy = SaliencyStats(saliency).region_centroid(20, 10, 60, 40)
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(30.0)

    def test_centroid_weighted_toward_heavier_mass(self):
        saliency = np.zeros((10, 100), dtype=np.float32)
        saliency[:, 10] = 3.0  # heavy column left
        saliency[:, 90] = 1.0  # light column right
        cx, _ = SaliencyStats(saliency).region_centroid(0, 0, 100, 10)
        assert cx == pytest.approx((3 * 10.5 + 1 * 90.5) / 4)

    def test_empty_region_returns_none(self):
        saliency = np.zeros((50, 50), dtype=np.float32)
        assert SaliencyStats(saliency).region_centroid(0, 0, 50, 50) is None
