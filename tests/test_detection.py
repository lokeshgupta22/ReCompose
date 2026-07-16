"""Tests for YOLOv8 ONNX pre/post-processing.

These target the pure functions (letterbox, decode) against synthetic
tensors with known expected outputs - no model download, no onnxruntime
session. The InferenceSession wrapper itself stays thin and untested here,
same policy as SaliencyEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from recompose.perception.detection import (
    YOLO_INPUT_SIZE,
    decode_predictions,
    letterbox,
)


def _blank(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _synthetic_output(rows: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    """Build a (1, 84, N) YOLOv8 output tensor from (cx, cy, w, h, class_id, score) rows."""
    n = len(rows)
    out = np.zeros((1, 84, n), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(rows):
        out[0, 0:4, i] = (cx, cy, w, h)
        out[0, 4 + class_id, i] = score
    return out


class TestLetterbox:
    def test_landscape_scales_to_width_and_pads_height(self):
        padded, scale, (pad_x, pad_y) = letterbox(_blank(1280, 720))
        assert padded.shape == (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3)
        assert scale == 0.5  # 1280 -> 640 is the binding dimension
        assert pad_x == 0
        assert pad_y == (YOLO_INPUT_SIZE - 360) / 2  # 720 * 0.5 = 360, centered

    def test_portrait_scales_to_height_and_pads_width(self):
        padded, scale, (pad_x, pad_y) = letterbox(_blank(720, 1280))
        assert scale == 0.5
        assert pad_y == 0
        assert pad_x == (YOLO_INPUT_SIZE - 360) / 2

    def test_padding_uses_yolo_gray(self):
        padded, _, (_, pad_y) = letterbox(_blank(1280, 720))
        # Top padding rows should be the 114-gray YOLO was trained with.
        assert (padded[0] == 114).all()
        # Image content region should be untouched (zeros).
        assert (padded[int(pad_y) + 1] == 0).all()

    def test_square_image_has_no_padding(self):
        padded, scale, (pad_x, pad_y) = letterbox(_blank(320, 320))
        assert padded.shape == (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3)
        assert scale == 2.0  # small images are upscaled, matching ultralytics
        assert pad_x == 0 and pad_y == 0


class TestDecodePredictions:
    def test_confident_subject_box_maps_back_to_original_pixels(self):
        # Original image 1280x720 letterboxed: scale 0.5, pad_y 140.
        # A person (class 0) centered at (320, 320) in 640-space, 100x200 box.
        output = _synthetic_output([(320.0, 320.0, 100.0, 200.0, 0, 0.9)])
        boxes = decode_predictions(
            output, scale=0.5, padding=(0.0, 140.0), image_size=(1280, 720), conf=0.4
        )
        assert len(boxes) == 1
        box = boxes[0]
        assert box.label == "person"
        # approx: the tensor is float32, and float32(0.9) != 0.9 exactly
        assert box.confidence == pytest.approx(0.9)
        # x: (320 - 50 - 0) / 0.5 = 540 ... (320 + 50) / 0.5 = 740
        assert box.x1 == 540 and box.x2 == 740
        # y: (320 - 100 - 140) / 0.5 = 160 ... (320 + 100 - 140) / 0.5 = 560
        assert box.y1 == 160 and box.y2 == 560

    def test_below_threshold_dropped(self):
        output = _synthetic_output([(320.0, 320.0, 100.0, 100.0, 0, 0.3)])
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert boxes == []

    def test_non_subject_class_dropped(self):
        # Class 9 is "traffic light" - detectable, but never a photographic subject.
        output = _synthetic_output([(320.0, 320.0, 100.0, 100.0, 9, 0.95)])
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert boxes == []

    def test_nms_collapses_overlapping_boxes_of_same_class(self):
        near_duplicates = [
            (320.0, 320.0, 100.0, 100.0, 0, 0.9),
            (322.0, 318.0, 100.0, 100.0, 0, 0.8),  # ~same box, lower score
        ]
        output = _synthetic_output(near_duplicates)
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert len(boxes) == 1
        assert boxes[0].confidence == pytest.approx(0.9)  # the higher-scoring one survives

    def test_nms_keeps_overlapping_boxes_of_different_classes(self):
        # A person holding a cat: heavy overlap, different classes, keep both.
        overlapping = [
            (320.0, 320.0, 100.0, 100.0, 0, 0.9),  # person
            (322.0, 318.0, 100.0, 100.0, 15, 0.8),  # cat
        ]
        output = _synthetic_output(overlapping)
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert {b.label for b in boxes} == {"person", "cat"}

    def test_boxes_clipped_to_image_bounds(self):
        # Box hanging off the left edge after unpadding must clip to x1 = 0.
        output = _synthetic_output([(10.0, 320.0, 100.0, 100.0, 0, 0.9)])
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert len(boxes) == 1
        assert boxes[0].x1 == 0

    def test_distant_boxes_of_same_class_both_survive(self):
        two_people = [
            (100.0, 100.0, 80.0, 80.0, 0, 0.9),
            (500.0, 500.0, 80.0, 80.0, 0, 0.85),
        ]
        output = _synthetic_output(two_people)
        boxes = decode_predictions(
            output, scale=1.0, padding=(0.0, 0.0), image_size=(640, 640), conf=0.4
        )
        assert len(boxes) == 2
