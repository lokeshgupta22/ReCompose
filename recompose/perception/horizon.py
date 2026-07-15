"""Horizon tilt estimation from near-horizontal line segments."""

from __future__ import annotations

import numpy as np


def estimate_horizon_tilt(
    image: np.ndarray,
    max_tilt_deg: float = 20.0,
    min_line_frac: float = 0.2,
) -> float | None:
    """Estimate the tilt of the horizon in degrees (positive = clockwise).

    Detects long, near-horizontal line segments and returns their
    length-weighted median angle. Returns None when no reliable
    horizontal structure is found (e.g. portraits, busy scenes).
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(w * min_line_frac),
        maxLineGap=10,
    )
    if lines is None:
        return None

    angles: list[float] = []
    weights: list[float] = []
    # HoughLinesP output shape drifts across cv2 versions ((N,1,4) vs (N,4));
    # normalize instead of indexing an assumed axis.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Fold into [-90, 90) so direction of the segment doesn't matter.
        if angle >= 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= max_tilt_deg:
            angles.append(angle)
            weights.append(length)

    if not angles:
        return None

    # Length-weighted median: long lines (actual horizon) dominate.
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    median_idx = int(np.searchsorted(cumulative, cumulative[-1] / 2))
    return float(sorted_angles[median_idx])
