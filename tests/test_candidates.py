"""Contract for the grid-anchor candidate generator.

The generator is pure geometry: given image dimensions and a target aspect,
produce a small set of meaningfully-different crop windows. No ML anywhere.
"""

import pytest

from recompose.cropping.candidates import (
    CandidateGridConfig,
    generate_candidates,
    parse_aspect,
)

IMAGE_SIZES = [(4000, 3000), (3000, 4000), (1920, 1080), (500, 500)]
ASPECTS = ["original", "1:1", "4:5", "16:9"]


class TestParseAspect:
    def test_original_means_no_fixed_ratio(self):
        assert parse_aspect("original") is None

    @pytest.mark.parametrize(
        ("aspect", "expected"),
        [("1:1", 1.0), ("4:5", 0.8), ("16:9", 16 / 9)],
    )
    def test_ratio_strings(self, aspect, expected):
        assert parse_aspect(aspect) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["", "4x5", "0:5", "4:0", "-4:5", "4:5:6"])
    def test_malformed_aspect_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_aspect(bad)


class TestGenerateCandidates:
    @pytest.mark.parametrize(("width", "height"), IMAGE_SIZES)
    @pytest.mark.parametrize("aspect", ASPECTS)
    def test_all_candidates_stay_within_image(self, width, height, aspect):
        for crop in generate_candidates(width, height, aspect):
            assert crop.x >= 0
            assert crop.y >= 0
            assert crop.x2 <= width
            assert crop.y2 <= height
            assert crop.w > 0
            assert crop.h > 0

    @pytest.mark.parametrize(("width", "height"), IMAGE_SIZES)
    @pytest.mark.parametrize("aspect", ["1:1", "4:5", "16:9"])
    def test_candidates_match_requested_ratio(self, width, height, aspect):
        ratio = parse_aspect(aspect)
        for crop in generate_candidates(width, height, aspect):
            assert crop.w / crop.h == pytest.approx(ratio, rel=0.02)

    @pytest.mark.parametrize(("width", "height"), IMAGE_SIZES)
    def test_original_aspect_preserves_image_ratio(self, width, height):
        for crop in generate_candidates(width, height, "original"):
            assert crop.w / crop.h == pytest.approx(width / height, rel=0.02)

    def test_no_duplicate_candidates(self):
        crops = generate_candidates(4000, 3000, "1:1")
        assert len(crops) == len(set(crops))

    def test_full_frame_is_a_candidate_for_original_aspect(self):
        crops = generate_candidates(4000, 3000, "original")
        assert any(c.w == 4000 and c.h == 3000 for c in crops)

    def test_candidate_count_bounded_by_config(self):
        config = CandidateGridConfig(scales=(0.55, 0.75, 1.0), positions_per_axis=3)
        crops = generate_candidates(4000, 3000, "1:1", config)
        assert 0 < len(crops) <= 3 * 3 * 3

    def test_no_crop_smaller_than_min_scale(self):
        config = CandidateGridConfig()
        min_scale = min(config.scales)
        for crop in generate_candidates(4000, 3000, "1:1", config):
            assert crop.w >= min_scale * 3000 * 0.99  # max 1:1 crop is 3000x3000

    def test_candidates_are_tagged_with_their_aspect(self):
        for crop in generate_candidates(1920, 1080, "4:5"):
            assert crop.aspect == "4:5"

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-5, 100)])
    def test_degenerate_image_dimensions_rejected(self, width, height):
        with pytest.raises(ValueError):
            generate_candidates(width, height, "1:1")
