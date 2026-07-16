"""Tests for GAICD batching: resize/normalize transforms and the collate_fn."""

import pytest
import torch

from training.batching import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NormalizeSample,
    ResizeSample,
    collate_crops,
)


def _sample(width, height, boxes, image_id="x"):
    return {
        "image": torch.zeros(3, height, width, dtype=torch.uint8),
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.full((len(boxes),), 3.0),
        "image_id": image_id,
    }


class TestResizeSample:
    def test_image_resized_to_square_float(self):
        out = ResizeSample(256)(_sample(1024, 512, [[0, 0, 100, 100]]))
        assert out["image"].shape == (3, 256, 256)
        assert out["image"].dtype == torch.float32

    def test_boxes_scale_per_axis_with_the_pixels(self):
        # 1024x512 -> 256x256: x shrinks by 4, y shrinks by 2. The resize is
        # aspect-distorting on purpose (GAIC protocol); boxes must distort
        # identically or they stop pointing at the same content.
        out = ResizeSample(256)(_sample(1024, 512, [[100, 100, 500, 300]]))
        assert torch.allclose(out["boxes"][0], torch.tensor([25.0, 50.0, 125.0, 150.0]))

    def test_scores_and_id_untouched(self):
        out = ResizeSample(64)(_sample(128, 128, [[0, 0, 10, 10]], image_id="keep"))
        assert out["image_id"] == "keep"
        assert out["scores"].tolist() == [3.0]


class TestNormalizeSample:
    def test_applies_imagenet_statistics_channelwise(self):
        sample = _sample(8, 8, [[0, 0, 4, 4]])
        sample["image"] = torch.ones(3, 8, 8)  # pretend already-resized floats in [0,1]
        out = NormalizeSample()(sample)
        expected = (1.0 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        assert torch.allclose(out["image"][:, 0, 0], expected, atol=1e-6)


class TestCollateCrops:
    def test_stacks_images_and_keeps_boxes_per_image(self):
        a = _sample(64, 64, [[0, 0, 10, 10], [5, 5, 20, 20]], image_id="a")
        b = _sample(64, 64, [[1, 2, 3, 4]], image_id="b")
        batch = collate_crops([a, b])
        assert batch["images"].shape == (2, 3, 64, 64)
        assert [len(bx) for bx in batch["boxes"]] == [2, 1]  # list form for roi_align
        assert batch["scores"].shape == (3,)  # concatenated across the batch
        assert batch["image_ids"] == ["a", "b"]

    def test_scores_concatenate_in_image_order(self):
        a = _sample(8, 8, [[0, 0, 1, 1]])
        a["scores"] = torch.tensor([1.0])
        b = _sample(8, 8, [[0, 0, 1, 1], [1, 1, 2, 2]])
        b["scores"] = torch.tensor([2.0, 3.0])
        batch = collate_crops([a, b])
        assert batch["scores"].tolist() == [1.0, 2.0, 3.0]

    def test_mismatched_image_sizes_rejected_clearly(self):
        with pytest.raises(ValueError, match="same spatial size"):
            collate_crops([_sample(64, 64, [[0, 0, 1, 1]]), _sample(32, 32, [[0, 0, 1, 1]])])
