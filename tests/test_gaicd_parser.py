"""Tests for GAICD annotation parsing - pure text in, crops out, no I/O."""

import pytest

from training.gaicd import CropAnnotation, parse_annotation_text

SAMPLE = """\
24 43 408 811 2.14
24 43 456 725 2.57
10 10 200 300 -2.00
24 43 456 896 4.86
"""


def test_parses_boxes_and_scores_in_file_order():
    crops = parse_annotation_text(SAMPLE)
    assert crops[0] == CropAnnotation(24, 43, 408, 811, 2.14)
    assert crops[-1] == CropAnnotation(24, 43, 456, 896, 4.86)


def test_unrated_sentinel_is_filtered_not_kept_as_low_score():
    # GAICD marks crops nobody rated with -2.0; a missing rating must never
    # be mistaken for a terrible one (same principle as Phase 1's None != 0).
    crops = parse_annotation_text(SAMPLE)
    assert len(crops) == 3
    assert all(c.score >= 0 for c in crops)


def test_blank_lines_and_trailing_whitespace_tolerated():
    assert len(parse_annotation_text("\n1 2 3 4 5.0  \n\n")) == 1


def test_empty_text_gives_empty_list():
    assert parse_annotation_text("") == []


def test_wrong_field_count_raises_with_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_annotation_text("1 2 3 4 5.0\n1 2 3 4\n")


def test_non_numeric_field_raises_with_line_number():
    with pytest.raises(ValueError, match="line 1"):
        parse_annotation_text("1 2 three 4 5.0\n")
