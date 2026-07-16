"""Tests for the CropScorer model - fake backbone, no pretrained downloads."""

import torch
from torch import nn

from training.model import CropScorer


class FakeBackbone(nn.Module):
    """Stride-4, 8-channel stand-in for a real CNN."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, images):
        return self.conv(images)


def _model() -> CropScorer:
    torch.manual_seed(7)
    return CropScorer(FakeBackbone(), feature_channels=8, stride=4, pool_size=3, hidden=16)


def test_one_score_per_box_across_variable_counts():
    model = _model()
    images = torch.rand(2, 3, 32, 32)
    boxes = [
        torch.tensor([[0.0, 0.0, 16.0, 16.0], [8.0, 8.0, 30.0, 30.0], [4.0, 0.0, 20.0, 12.0]]),
        torch.tensor([[2.0, 2.0, 28.0, 28.0]]),
    ]
    scores = model(images, boxes)
    assert scores.shape == (4,)


def test_gradients_flow_to_the_backbone():
    model = _model()
    images = torch.rand(1, 3, 32, 32)
    scores = model(images, [torch.tensor([[0.0, 0.0, 16.0, 16.0]])])
    scores.sum().backward()
    grad = model.backbone.conv.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_different_boxes_score_differently():
    model = _model()
    images = torch.rand(1, 3, 32, 32)
    boxes = [torch.tensor([[0.0, 0.0, 8.0, 8.0], [16.0, 16.0, 32.0, 32.0]])]
    scores = model(images, boxes)
    assert scores[0] != scores[1]


def test_each_box_reads_its_own_images_features():
    model = _model()
    # Identical box on two very different images must score differently -
    # proving roi_align indexes the right batch element.
    images = torch.cat([torch.zeros(1, 3, 32, 32), torch.rand(1, 3, 32, 32)])
    box = torch.tensor([[4.0, 4.0, 28.0, 28.0]])
    scores = model(images, [box, box.clone()])
    assert scores[0] != scores[1]


def test_eval_mode_is_deterministic():
    model = _model().eval()
    images = torch.rand(1, 3, 32, 32)
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    with torch.no_grad():
        assert torch.equal(model(images, boxes), model(images, boxes))
