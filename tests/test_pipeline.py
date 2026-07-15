"""Contract for the end-to-end analysis pipeline.

Perception models are injected (dependency inversion), so these tests run on
synthetic fakes - no model downloads, no GPU, sub-second CI.
"""

import numpy as np
import pytest

from recompose.pipeline import RecomposePipeline
from recompose.types import Box


class FakeSaliency:
    def __init__(self, saliency):
        self._saliency = saliency

    def predict(self, image):
        return self._saliency


class FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, image):
        return self._boxes


def make_pipeline(saliency, boxes, tilt=-2.0):
    return RecomposePipeline(
        saliency_model=FakeSaliency(saliency),
        subject_model=FakeDetector(boxes),
        horizon_fn=lambda image: tilt,
    )


@pytest.fixture
def dog_scene():
    """400x300 scene: saliency blob and a dog box around it."""
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    saliency = np.zeros((300, 400), dtype=np.float32)
    saliency[100:180, 120:220] = 1.0
    dog = Box(130, 110, 210, 170, label="dog", confidence=0.9)
    return image, saliency, dog


class TestAnalyze:
    def test_returns_exactly_the_requested_aspects(self, dog_scene):
        image, saliency, dog = dog_scene
        result = make_pipeline(saliency, [dog]).analyze(image, aspects=("original", "1:1"))
        assert set(result.crops_by_aspect) == {"original", "1:1"}

    def test_crops_ranked_best_first_with_scores_in_unit_interval(self, dog_scene):
        image, saliency, dog = dog_scene
        result = make_pipeline(saliency, [dog]).analyze(image, aspects=("1:1",), top_k=3)
        scored = result.crops_by_aspect["1:1"]
        assert 0 < len(scored) <= 3
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_returned_crops_never_cut_the_subject(self, dog_scene):
        image, saliency, dog = dog_scene
        result = make_pipeline(saliency, [dog]).analyze(image)
        for scored_list in result.crops_by_aspect.values():
            for scored in scored_list:
                overlap = dog.intersection_area(scored.crop.as_box()) / dog.area
                assert overlap == pytest.approx(1.0)

    def test_result_carries_scene_metadata(self, dog_scene):
        image, saliency, dog = dog_scene
        result = make_pipeline(saliency, [dog], tilt=-2.0).analyze(image)
        assert (result.width, result.height) == (400, 300)
        assert result.horizon_tilt_deg == pytest.approx(-2.0)
        assert result.subjects == [dog]
        assert result.saliency.shape == (300, 400)
        assert result.constraints_relaxed is False

    def test_relaxes_constraints_rather_than_returning_nothing(self):
        # Panorama: no 1:1 crop can retain 55% of uniform saliency.
        image = np.zeros((300, 1000, 3), dtype=np.uint8)
        saliency = np.full((300, 1000), 0.5, dtype=np.float32)
        result = make_pipeline(saliency, []).analyze(image, aspects=("1:1",))
        assert len(result.crops_by_aspect["1:1"]) > 0
        assert result.constraints_relaxed is True

    def test_rejects_non_rgb_input(self, dog_scene):
        _, saliency, dog = dog_scene
        with pytest.raises(ValueError):
            make_pipeline(saliency, [dog]).analyze(np.zeros((300, 400), dtype=np.uint8))

    def test_rejects_saliency_shape_mismatch(self, dog_scene):
        image, _, dog = dog_scene
        wrong = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            make_pipeline(wrong, [dog]).analyze(image)
