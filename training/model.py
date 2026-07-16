"""GAIC-style crop scorer: shared backbone, RoIAlign, discard-aware head.

The backbone runs once per image; every candidate crop is scored from the
shared feature map (the Fast R-CNN trick - candidates overlap so heavily
that per-crop forward passes would recompute ~90x redundant work).

Each crop's representation concatenates:
- RoIAlign features of the crop itself (what the framing keeps), and
- the per-channel mean of features OUTSIDE the crop (what it throws away).

The second term is a lightweight stand-in for the GAIC paper's RoDAlign:
composition quality depends on the discarded content too - cutting empty
sky is different from cutting half a face. Phase 1 encoded the same idea
as the hand-written retention rule.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import roi_align


class CropScorer(nn.Module):
    """Scores candidate crops; higher = better composition."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: int,
        stride: int,
        pool_size: int = 9,
        hidden: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.stride = stride
        self.pool_size = pool_size
        head_in = feature_channels * pool_size * pool_size + feature_channels
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, images: torch.Tensor, boxes: list[torch.Tensor]) -> torch.Tensor:
        """images: (B, 3, H, W); boxes: per-image (N_i, 4) pixel xyxy.

        Returns (sum N_i,) scores, ordered image 0's boxes first, then 1's.
        """
        fmap = self.backbone(images)  # (B, C, H/stride, W/stride)
        rois = roi_align(
            fmap,
            boxes,
            output_size=self.pool_size,
            spatial_scale=1.0 / self.stride,
            aligned=True,
        )  # (K, C, pool, pool)

        inside = rois.flatten(start_dim=1)
        outside_mean = self._outside_mean(fmap, boxes, rois)
        return self.head(torch.cat([inside, outside_mean], dim=1)).squeeze(-1)

    def _outside_mean(
        self,
        fmap: torch.Tensor,
        boxes: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> torch.Tensor:
        """Per-channel mean of feature cells outside each box, in closed form.

        outside_sum = image_total - inside_sum, where inside_sum is
        approximated as RoIAlign's mean times the box's cell count. Exact
        enough for a context signal, and entirely differentiable.
        """
        _, _, height, width = fmap.shape
        totals = fmap.sum(dim=(2, 3))  # (B, C)
        image_index = torch.repeat_interleave(
            torch.arange(len(boxes), device=fmap.device),
            torch.tensor([len(b) for b in boxes], device=fmap.device),
        )
        all_boxes = torch.cat(boxes) / self.stride  # feature-cell coordinates
        cells = ((all_boxes[:, 2] - all_boxes[:, 0]).clamp(min=1.0)
                 * (all_boxes[:, 3] - all_boxes[:, 1]).clamp(min=1.0)).unsqueeze(1)
        inside_sum = rois.mean(dim=(2, 3)) * cells
        outside_cells = (height * width - cells).clamp(min=1.0)
        return (totals[image_index] - inside_sum) / outside_cells
