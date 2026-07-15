"""Regression contract for horizon tilt estimation.

Runs the real OpenCV path on synthetic images, which pins the
HoughLinesP output-shape handling across cv2 versions (an (N,1,4) vs
(N,4) drift broke production while unit tests stayed green).
"""

import math

import cv2
import numpy as np
import pytest

from recompose.perception.horizon import estimate_horizon_tilt


def scene_with_line(tilt_deg: float, size=(400, 600)) -> np.ndarray:
    """Dark image with one long bright line at the given tilt."""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x1, x2 = 40, width - 40
    dy = math.tan(math.radians(tilt_deg)) * (x2 - x1)
    y1 = height // 2 - int(dy / 2)
    y2 = height // 2 + int(dy / 2)
    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
    return image


class TestEstimateHorizonTilt:
    @pytest.mark.parametrize("tilt", [0.0, 3.0, -5.0])
    def test_recovers_known_tilt(self, tilt):
        estimate = estimate_horizon_tilt(scene_with_line(tilt))
        assert estimate is not None
        assert estimate == pytest.approx(tilt, abs=0.75)

    def test_blank_image_gives_no_opinion(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        assert estimate_horizon_tilt(image) is None

    def test_steep_lines_are_not_horizons(self):
        assert estimate_horizon_tilt(scene_with_line(45.0)) is None
