"""Tests for the GAICD torch Dataset - runs on a tiny synthetic fixture tree."""

import numpy as np
import pytest
import torch
from PIL import Image

from training.gaicd import GAICDataset

ANNOTATION_A = """\
10 20 60 40 3.5
0 0 32 24 -2.0
5 5 50 45 2.0
"""
ANNOTATION_B = "1 2 30 40 4.0\n"


@pytest.fixture
def gaicd_root(tmp_path):
    for split_dir in ("images/train", "annotations/train"):
        (tmp_path / split_dir).mkdir(parents=True)
    for stem, text, size in (("aaa", ANNOTATION_A, (64, 48)), ("bbb", ANNOTATION_B, (32, 32))):
        Image.new("RGB", size, color=(120, 30, 200)).save(tmp_path / "images/train" / f"{stem}.jpg")
        (tmp_path / "annotations/train" / f"{stem}.txt").write_text(text)
    return tmp_path


def test_length_is_number_of_annotation_files(gaicd_root):
    assert len(GAICDataset(gaicd_root, "train")) == 2


def test_sample_shapes_and_types(gaicd_root):
    sample = GAICDataset(gaicd_root, "train")[0]  # sorted -> "aaa"
    assert sample["image_id"] == "aaa"
    assert sample["image"].dtype == torch.uint8
    assert sample["image"].shape == (3, 48, 64)  # CHW from a 64x48 file
    assert sample["boxes"].dtype == torch.float32
    assert sample["boxes"].shape == (2, 4)  # 3 lines minus the -2.0 sentinel
    assert sample["scores"].shape == (2,)


def test_boxes_and_scores_match_annotation_order(gaicd_root):
    sample = GAICDataset(gaicd_root, "train")[0]
    assert torch.equal(sample["boxes"][0], torch.tensor([10.0, 20.0, 60.0, 40.0]))
    assert sample["scores"].tolist() == pytest.approx([3.5, 2.0])


def test_transform_is_applied_to_the_image(gaicd_root):
    dataset = GAICDataset(gaicd_root, "train", transform=lambda image: image.float() / 255.0)
    assert dataset[1]["image"].dtype == torch.float32
    assert dataset[1]["image"].max() <= 1.0


def test_unknown_split_rejected(gaicd_root):
    with pytest.raises(ValueError, match="dev"):
        GAICDataset(gaicd_root, "dev")


def test_empty_split_rejected(gaicd_root):
    (gaicd_root / "annotations/val").mkdir()
    (gaicd_root / "images/val").mkdir()
    with pytest.raises(FileNotFoundError, match="val"):
        GAICDataset(gaicd_root, "val")


def test_missing_image_named_in_error(gaicd_root):
    (gaicd_root / "images/train/aaa.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="aaa"):
        GAICDataset(gaicd_root, "train")[0]


def test_image_pixels_survive_the_round_trip(gaicd_root):
    image = GAICDataset(gaicd_root, "train")[0]["image"]
    # JPEG is lossy, so allow slack - but the solid color must come through.
    mean_rgb = image.float().mean(dim=(1, 2))
    assert np.allclose(mean_rgb.numpy(), [120, 30, 200], atol=6)
