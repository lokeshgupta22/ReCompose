"""Transforms and collate for GAICD training batches.

The resize is aspect-distorting to a fixed square (the GAIC protocol):
composition ratings were collected on the original framings, and the
model only ever compares crops within one image, so a consistent
distortion cancels out - while fixed-size inputs make batching trivial.

Normalization constants are ImageNet mean/std because the backbone is
ImageNet-pretrained. These constants are part of the model contract:
any future export must ship them (see Phase 1's saliency preprocessing
for the same pattern).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeSample:
    """Resize image to size x size and scale boxes identically, per axis."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        _, height, width = image.shape
        resized = F.interpolate(
            image.unsqueeze(0).float(),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        scale_x = self.size / width
        scale_y = self.size / height
        scale = torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return {**sample, "image": resized, "boxes": sample["boxes"] * scale}


class NormalizeSample:
    """Scale [0,255] floats to [0,1] if needed, then apply ImageNet mean/std."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"].float()
        if image.max() > 1.5:  # still in [0,255]
            image = image / 255.0
        return {**sample, "image": (image - self._mean) / self._std}


class ComposeSample:
    """Apply sample-level transforms in order."""

    def __init__(self, *transforms):
        self._transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for transform in self._transforms:
            sample = transform(sample)
        return sample


def collate_crops(samples: list[dict]) -> dict:
    """Batch samples with a variable number of crops per image.

    Images stack into one (B, C, H, W) tensor; boxes stay a list of
    (N_i, 4) tensors - exactly the format torchvision's roi_align takes -
    and scores concatenate to (sum N_i,), matching roi_align's output
    ordering (all boxes of image 0, then image 1, ...).
    """
    shapes = {tuple(s["image"].shape[-2:]) for s in samples}
    if len(shapes) != 1:
        raise ValueError(
            f"all images in a batch must share the same spatial size, got {sorted(shapes)}; "
            "apply ResizeSample in the dataset transform"
        )
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "boxes": [s["boxes"] for s in samples],
        "scores": torch.cat([s["scores"] for s in samples]),
        "image_ids": [s["image_id"] for s in samples],
    }
