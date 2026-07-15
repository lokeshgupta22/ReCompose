"""Contract for crop constraint filtering.

Constraints are pluggable pass/fail judges (strategy objects). A crop survives
only if every constraint allows it. Synthetic saliency maps and boxes keep
these tests free of any ML model.
"""

import numpy as np
import pytest

from recompose.cropping.candidates import generate_candidates
from recompose.cropping.constraints import (
    SaliencyRetentionConstraint,
    SubjectIntegrityConstraint,
    filter_candidates,
)
from recompose.saliency_stats import SaliencyStats
from recompose.types import Box, CropCandidate


def crop(x, y, w, h):
    return CropCandidate(x=x, y=y, w=w, h=h)


class TestSaliencyStats:
    def test_region_sum_matches_numpy_slicing(self):
        rng = np.random.default_rng(seed=7)
        saliency = rng.random((60, 80)).astype(np.float32)
        stats = SaliencyStats(saliency)
        assert stats.region_sum(10, 5, 30, 40) == pytest.approx(
            float(saliency[5:45, 10:40].sum()), rel=1e-5
        )

    def test_total_is_whole_image_sum(self):
        saliency = np.ones((10, 10), dtype=np.float32)
        assert SaliencyStats(saliency).total == pytest.approx(100.0)

    def test_region_clipped_to_image_bounds(self):
        saliency = np.ones((10, 10), dtype=np.float32)
        stats = SaliencyStats(saliency)
        assert stats.region_sum(-5, -5, 20, 20) == pytest.approx(100.0)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            SaliencyStats(np.ones((10, 10, 3), dtype=np.float32))


class TestSubjectIntegrityConstraint:
    subject = Box(100, 100, 200, 200, label="person")

    def test_crop_containing_subject_is_allowed(self):
        constraint = SubjectIntegrityConstraint([self.subject])
        assert constraint.allows(crop(50, 50, 300, 300))

    def test_crop_cutting_subject_is_rejected(self):
        constraint = SubjectIntegrityConstraint([self.subject])
        assert not constraint.allows(crop(0, 0, 150, 300))  # slices subject vertically

    def test_primary_subject_must_be_included(self):
        constraint = SubjectIntegrityConstraint([self.subject])
        assert not constraint.allows(crop(300, 300, 100, 100))  # excludes subject

    def test_secondary_subject_may_be_fully_excluded(self):
        primary = Box(100, 100, 300, 300)  # largest -> primary
        secondary = Box(400, 400, 420, 420)
        constraint = SubjectIntegrityConstraint([primary, secondary])
        assert constraint.allows(crop(50, 50, 300, 300))

    def test_secondary_subject_may_not_be_cut(self):
        primary = Box(100, 100, 300, 300)
        secondary = Box(400, 400, 440, 440)
        constraint = SubjectIntegrityConstraint([primary, secondary])
        assert not constraint.allows(crop(50, 50, 370, 370))  # slices secondary

    def test_no_subjects_allows_everything(self):
        constraint = SubjectIntegrityConstraint([])
        assert constraint.allows(crop(0, 0, 10, 10))


class TestSaliencyRetentionConstraint:
    def make_stats(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        saliency[10:40, 10:40] = 1.0  # all mass in top-left quadrant
        return SaliencyStats(saliency)

    def test_crop_keeping_mass_is_allowed(self):
        constraint = SaliencyRetentionConstraint(self.make_stats(), min_retention=0.6)
        assert constraint.allows(crop(0, 0, 50, 50))

    def test_crop_abandoning_mass_is_rejected(self):
        constraint = SaliencyRetentionConstraint(self.make_stats(), min_retention=0.6)
        assert not constraint.allows(crop(50, 50, 50, 50))

    def test_empty_saliency_allows_everything(self):
        stats = SaliencyStats(np.zeros((100, 100), dtype=np.float32))
        constraint = SaliencyRetentionConstraint(stats, min_retention=0.6)
        assert constraint.allows(crop(90, 90, 10, 10))


class TestFilterCandidates:
    def test_only_crops_passing_all_constraints_survive(self):
        saliency = np.zeros((300, 400), dtype=np.float32)
        saliency[100:200, 100:200] = 1.0
        subjects = [Box(120, 120, 180, 180, label="dog")]
        candidates = generate_candidates(400, 300, "1:1")
        survivors = filter_candidates(
            candidates,
            [
                SubjectIntegrityConstraint(subjects),
                SaliencyRetentionConstraint(SaliencyStats(saliency), min_retention=0.6),
            ],
        )
        assert 0 < len(survivors) < len(candidates)
        subject_box = subjects[0]
        for c in survivors:
            frac = subject_box.intersection_area(c.as_box()) / subject_box.area
            assert frac == pytest.approx(1.0)

    def test_no_constraints_keeps_everything(self):
        candidates = generate_candidates(400, 300, "1:1")
        assert filter_candidates(candidates, []) == candidates
