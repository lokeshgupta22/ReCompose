"""Rule-based composition scoring.

Each rule maps a candidate crop to a score in [0, 1], or None when the rule
does not apply (e.g. headroom without subjects). The scorer fuses applicable
rules by weighted average, excluding inapplicable weights from the
denominator so scores degrade gracefully instead of collapsing.

This scorer is the hand-crafted baseline that the learned model (Phase 2)
must beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from recompose.saliency_stats import SaliencyStats
from recompose.types import Box, CropCandidate, ScoredCrop

# Sigma of one 3x3 grid cell: being a full cell away from a power point
# decays the thirds score to exp(-4) ~= 0.02.
THIRDS_SIGMA = 1 / 6
POWER_POINTS = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))

# A subject counts as "in the crop" for subject-dependent rules when at
# least this fraction of its box is visible.
VISIBLE_FRACTION = 0.5

DEFAULT_RULE_WEIGHTS = {
    "thirds": 0.40,
    "retention": 0.25,
    "balance": 0.15,
    "headroom": 0.10,
    "subject_size": 0.10,
}


class CompositionRule(Protocol):
    name: str

    def score(self, crop: CropCandidate) -> float | None: ...


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """1.0 on the plateau [b, c], sloping linearly to 0 at a and d.

    The plateau encodes honest uncertainty about the ideal value; a single
    peak would claim precision we don't have.
    """
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


@dataclass(frozen=True)
class ThirdsRule:
    """Gaussian falloff of the saliency centroid's distance (in crop-relative
    coordinates) to the nearest rule-of-thirds power point."""

    stats: SaliencyStats
    name: ClassVar[str] = "thirds"

    def score(self, crop: CropCandidate) -> float | None:
        centroid = self.stats.region_centroid(crop.x, crop.y, crop.w, crop.h)
        if centroid is None:
            return None
        u = (centroid[0] - crop.x) / crop.w
        v = (centroid[1] - crop.y) / crop.h
        distance = min(math.hypot(u - px, v - py) for px, py in POWER_POINTS)
        return math.exp(-((distance / THIRDS_SIGMA) ** 2))


@dataclass(frozen=True)
class BalanceRule:
    """Left/right visual-mass balance: 1 when halves match, 0 when one-sided."""

    stats: SaliencyStats
    name: ClassVar[str] = "balance"

    def score(self, crop: CropCandidate) -> float | None:
        half_w = crop.w // 2
        left = self.stats.region_sum(crop.x, crop.y, half_w, crop.h)
        right = self.stats.region_sum(crop.x + half_w, crop.y, crop.w - half_w, crop.h)
        total = left + right
        if total <= 0:
            return None
        return 1.0 - abs(left - right) / total


@dataclass(frozen=True)
class RetentionRule:
    """Fraction of the image's total visual mass the crop keeps."""

    stats: SaliencyStats
    name: ClassVar[str] = "retention"

    def score(self, crop: CropCandidate) -> float | None:
        if self.stats.total <= 0:
            return None
        return self.stats.region_sum(crop.x, crop.y, crop.w, crop.h) / self.stats.total


@dataclass(frozen=True)
class HeadroomRule:
    """Breathing room above the topmost visible subject: comfortable within
    4-18% of crop height, cramped at 0, drowning past 40%."""

    subjects: Sequence[Box]
    name: ClassVar[str] = "headroom"

    def score(self, crop: CropCandidate) -> float | None:
        visible = _visible_subjects(self.subjects, crop)
        if not visible:
            return None
        topmost = min(box.y1 for box in visible)
        headroom = max(0.0, (topmost - crop.y) / crop.h)
        return _trapezoid(headroom, 0.0, 0.04, 0.18, 0.40)


@dataclass(frozen=True)
class SubjectSizeRule:
    """Primary subject's share of the crop area: healthy within 8-50%,
    a speck below 3%, wall-filling above 75%."""

    subjects: Sequence[Box]
    name: ClassVar[str] = "subject_size"

    def score(self, crop: CropCandidate) -> float | None:
        visible = _visible_subjects(self.subjects, crop)
        if not visible:
            return None
        primary = max(visible, key=lambda box: box.area)
        fraction = primary.intersection_area(crop.as_box()) / crop.as_box().area
        return _trapezoid(fraction, 0.03, 0.08, 0.50, 0.75)


def _visible_subjects(subjects: Sequence[Box], crop: CropCandidate) -> list[Box]:
    crop_box = crop.as_box()
    return [
        box
        for box in subjects
        if box.area > 0 and box.intersection_area(crop_box) / box.area >= VISIBLE_FRACTION
    ]


class CompositionScorer:
    """Weighted-average fusion of composition rules."""

    def __init__(self, weighted_rules: Sequence[tuple[CompositionRule, float]]):
        if not weighted_rules:
            raise ValueError("scorer needs at least one rule")
        self._weighted_rules = list(weighted_rules)

    def score(self, crop: CropCandidate) -> ScoredCrop:
        rule_scores: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        for rule, weight in self._weighted_rules:
            value = rule.score(crop)
            if value is None:
                continue
            rule_scores[rule.name] = value
            weighted_sum += weight * value
            weight_total += weight
        overall = weighted_sum / weight_total if weight_total > 0 else 0.5
        return ScoredCrop(crop=crop, score=overall, rule_scores=rule_scores)


def build_default_scorer(stats: SaliencyStats, subjects: Sequence[Box]) -> CompositionScorer:
    rules: list[CompositionRule] = [
        ThirdsRule(stats),
        RetentionRule(stats),
        BalanceRule(stats),
        HeadroomRule(subjects),
        SubjectSizeRule(subjects),
    ]
    return CompositionScorer([(rule, DEFAULT_RULE_WEIGHTS[rule.name]) for rule in rules])
