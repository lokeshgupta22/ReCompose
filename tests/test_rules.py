"""Contract for the rule-based composition scorer.

Each rule maps a candidate crop to [0, 1] or None when inapplicable.
The scorer fuses applicable rules by weighted average.
"""

import numpy as np
import pytest

from recompose.saliency_stats import SaliencyStats
from recompose.scoring.rules import (
    BalanceRule,
    CompositionScorer,
    HeadroomRule,
    RetentionRule,
    SubjectSizeRule,
    ThirdsRule,
    build_default_scorer,
)
from recompose.types import Box, CropCandidate


def crop(x, y, w, h):
    return CropCandidate(x=x, y=y, w=w, h=h)


def blob_stats():
    """A single 20x20 unit blob centred at (150, 100) in a 300x200 image."""
    saliency = np.zeros((200, 300), dtype=np.float32)
    saliency[90:110, 140:160] = 1.0
    return SaliencyStats(saliency)


class TestThirdsRule:
    def test_centroid_on_power_point_beats_centered(self):
        rule = ThirdsRule(blob_stats())
        # 150x150 crop with the blob centroid at its (1/3, 1/3) power point:
        on_thirds = rule.score(crop(100, 50, 150, 150))
        centered = rule.score(crop(75, 25, 150, 150))
        assert on_thirds > 0.9
        assert centered < 0.3
        assert on_thirds > centered

    def test_scores_stay_in_unit_interval(self):
        rule = ThirdsRule(blob_stats())
        for c in [crop(0, 0, 300, 200), crop(100, 50, 150, 150), crop(140, 90, 40, 40)]:
            assert 0.0 <= rule.score(c) <= 1.0

    def test_massless_crop_is_inapplicable(self):
        rule = ThirdsRule(blob_stats())
        assert rule.score(crop(200, 150, 50, 50)) is None


class TestBalanceRule:
    def test_symmetric_mass_scores_high(self):
        saliency = np.zeros((200, 300), dtype=np.float32)
        saliency[90:110, 60:80] = 1.0
        saliency[90:110, 220:240] = 1.0
        rule = BalanceRule(SaliencyStats(saliency))
        assert rule.score(crop(0, 0, 300, 200)) == pytest.approx(1.0)

    def test_one_sided_mass_scores_low(self):
        saliency = np.zeros((200, 300), dtype=np.float32)
        saliency[90:110, 10:30] = 1.0
        rule = BalanceRule(SaliencyStats(saliency))
        assert rule.score(crop(0, 0, 300, 200)) == pytest.approx(0.0)

    def test_massless_crop_is_inapplicable(self):
        rule = BalanceRule(blob_stats())
        assert rule.score(crop(200, 150, 50, 50)) is None


class TestRetentionRule:
    def test_keeping_everything_scores_one(self):
        rule = RetentionRule(blob_stats())
        assert rule.score(crop(0, 0, 300, 200)) == pytest.approx(1.0)

    def test_missing_everything_scores_zero(self):
        rule = RetentionRule(blob_stats())
        assert rule.score(crop(200, 150, 50, 50)) == pytest.approx(0.0)

    def test_empty_saliency_is_inapplicable(self):
        stats = SaliencyStats(np.zeros((10, 10), dtype=np.float32))
        assert RetentionRule(stats).score(crop(0, 0, 10, 10)) is None


class TestHeadroomRule:
    subject = Box(100, 60, 200, 160, label="person")

    def test_comfortable_headroom_scores_full(self):
        rule = HeadroomRule([self.subject])
        assert rule.score(crop(50, 50, 200, 200)) == pytest.approx(1.0)  # 5% headroom

    def test_zero_headroom_scores_zero(self):
        rule = HeadroomRule([self.subject])
        assert rule.score(crop(50, 60, 200, 200)) == pytest.approx(0.0)

    def test_excessive_headroom_scores_zero(self):
        rule = HeadroomRule([self.subject])
        assert rule.score(crop(0, 0, 300, 120)) == pytest.approx(0.0)  # 50% headroom

    def test_no_subjects_is_inapplicable(self):
        assert HeadroomRule([]).score(crop(0, 0, 100, 100)) is None

    def test_subject_outside_crop_is_inapplicable(self):
        rule = HeadroomRule([self.subject])
        assert rule.score(crop(0, 0, 50, 50)) is None


class TestSubjectSizeRule:
    def test_healthy_subject_fraction_scores_full(self):
        rule = SubjectSizeRule([Box(50, 50, 150, 150)])  # 100x100 in 200x200 = 25%
        assert rule.score(crop(0, 0, 200, 200)) == pytest.approx(1.0)

    def test_speck_subject_scores_zero(self):
        rule = SubjectSizeRule([Box(0, 0, 10, 10)])
        assert rule.score(crop(0, 0, 2000, 2000)) == pytest.approx(0.0)

    def test_no_subjects_is_inapplicable(self):
        assert SubjectSizeRule([]).score(crop(0, 0, 100, 100)) is None


class _ConstantRule:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def score(self, _crop):
        return self._value


class TestCompositionScorer:
    def test_weighted_average_of_applicable_rules(self):
        scorer = CompositionScorer([(_ConstantRule("a", 1.0), 3.0), (_ConstantRule("b", 0.0), 1.0)])
        scored = scorer.score(crop(0, 0, 10, 10))
        assert scored.score == pytest.approx(0.75)
        assert scored.rule_scores == {"a": 1.0, "b": 0.0}

    def test_inapplicable_rules_excluded_from_denominator(self):
        scorer = CompositionScorer(
            [(_ConstantRule("a", 0.8), 1.0), (_ConstantRule("none", None), 100.0)]
        )
        scored = scorer.score(crop(0, 0, 10, 10))
        assert scored.score == pytest.approx(0.8)
        assert "none" not in scored.rule_scores

    def test_all_inapplicable_gives_neutral_score(self):
        scorer = CompositionScorer([(_ConstantRule("none", None), 1.0)])
        assert scorer.score(crop(0, 0, 10, 10)).score == pytest.approx(0.5)

    def test_rejects_empty_rule_list(self):
        with pytest.raises(ValueError):
            CompositionScorer([])


class TestBuildDefaultScorer:
    def test_scores_full_frame_with_all_rules_applicable(self):
        subjects = [Box(140, 90, 160, 110, label="cat")]
        scorer = build_default_scorer(blob_stats(), subjects)
        scored = scorer.score(crop(0, 0, 300, 200))
        assert 0.0 <= scored.score <= 1.0
        expected = {"thirds", "retention", "balance", "headroom", "subject_size"}
        assert set(scored.rule_scores) == expected

    def test_without_subjects_only_saliency_rules_apply(self):
        scorer = build_default_scorer(blob_stats(), [])
        scored = scorer.score(crop(0, 0, 300, 200))
        assert set(scored.rule_scores) == {"thirds", "retention", "balance"}
