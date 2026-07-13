"""Shared datatypes for the ReCompose pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: Box) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


@dataclass(frozen=True)
class CropCandidate:
    """Candidate crop window in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    aspect: str = "original"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def as_box(self) -> Box:
        return Box(self.x, self.y, self.x2, self.y2)


@dataclass
class PerceptionResult:
    """Everything the perception stage extracts from an image."""

    saliency: np.ndarray  # HxW float32 in [0, 1]
    subjects: list[Box] = field(default_factory=list)
    horizon_tilt_deg: float | None = None


@dataclass
class ScoredCrop:
    """A candidate crop with its composition scores."""

    crop: CropCandidate
    score: float
    rule_scores: dict[str, float] = field(default_factory=dict)
