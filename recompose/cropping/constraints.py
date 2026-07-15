"""Hard constraints: pass/fail judges that prune candidate crops.

Each constraint is a small strategy object with a single `allows` method, so
the filter is open for extension (add a constraint class) and closed for
modification (`filter_candidates` never changes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from recompose.saliency_stats import SaliencyStats
from recompose.types import Box, CropCandidate


class CropConstraint(Protocol):
    def allows(self, crop: CropCandidate) -> bool: ...


@dataclass(frozen=True)
class SubjectIntegrityConstraint:
    """No crop may slice through a subject, and the primary subject (largest
    box) must be kept. Secondary subjects may be fully excluded - dropping a
    background passer-by is legitimate recomposition, cutting them in half
    is not.

    A box counts as contained above `contained_threshold` overlap and as
    excluded below `excluded_threshold`; between the two it is being cut.
    The tolerance absorbs detector-box jitter at crop edges.
    """

    subjects: Sequence[Box]
    contained_threshold: float = 0.95
    excluded_threshold: float = 0.05

    def allows(self, crop: CropCandidate) -> bool:
        if not self.subjects:
            return True
        crop_box = crop.as_box()
        primary = max(self.subjects, key=lambda box: box.area)
        for subject in self.subjects:
            if subject.area <= 0:
                continue
            overlap = subject.intersection_area(crop_box) / subject.area
            if self.excluded_threshold < overlap < self.contained_threshold:
                return False
            if subject is primary and overlap < self.contained_threshold:
                return False
        return True


@dataclass(frozen=True)
class SaliencyRetentionConstraint:
    """A crop must keep at least `min_retention` of the image's total visual
    mass. An empty map means no evidence, so nothing is rejected."""

    stats: SaliencyStats
    min_retention: float = 0.55
    _EPSILON = 1e-6

    def allows(self, crop: CropCandidate) -> bool:
        if self.stats.total <= self._EPSILON:
            return True
        kept = self.stats.region_sum(crop.x, crop.y, crop.w, crop.h)
        return kept / self.stats.total >= self.min_retention


def filter_candidates(
    candidates: Sequence[CropCandidate],
    constraints: Sequence[CropConstraint],
) -> list[CropCandidate]:
    return [c for c in candidates if all(k.allows(c) for k in constraints)]
