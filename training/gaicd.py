"""GAICD (Grid Anchor based Image Cropping Database) annotation parsing.

Lives in the top-level `training` package on purpose: the deployed Docker
image copies only `recompose/` and `api/`, so training code structurally
cannot enter (or bloat) the 512MB runtime.

Annotation format (extended TPAMI release): one .txt per image, one line
per candidate crop - `x1 y1 x2 y2 mean_score` in pixel coordinates, with
human ratings on a 1-5 scale. A score of -2.0 marks a crop nobody rated;
those are filtered here, because a missing rating is not a low rating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CropAnnotation:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def parse_annotation_text(text: str) -> list[CropAnnotation]:
    crops: list[CropAnnotation] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"line {lineno}: expected 5 fields (x1 y1 x2 y2 score), "
                f"got {len(parts)}: {line!r}"
            )
        try:
            x1, y1, x2, y2, score = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"line {lineno}: non-numeric field in {line!r}") from exc
        if score < 0:
            continue  # unrated sentinel (-2.0), not a genuinely bad crop
        crops.append(CropAnnotation(x1, y1, x2, y2, score))
    return crops


class GAICDataset:
    """One sample per image: the pixels plus every rated candidate crop.

    Implements the torch Dataset protocol (__len__/__getitem__) without
    inheriting from torch.utils.data.Dataset - the base class adds nothing
    but an isinstance tag, and duck typing keeps this module importable
    for parsing-only use without torch installed.

    Samples are dicts: image (uint8 CHW tensor), boxes ((N, 4) float32,
    pixel xyxy), scores ((N,) float32, 1-5 human means), image_id (stem).
    Variable N per image means batching needs a custom collate_fn; that
    belongs to the training loop, not the dataset.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        transform: Callable | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self._image_dir = root / "images" / split
        self._annotation_files = sorted((root / "annotations" / split).glob("*.txt"))
        if not self._annotation_files:
            raise FileNotFoundError(f"no annotation files under {root / 'annotations' / split}")
        self._transform = transform

    def __len__(self) -> int:
        return len(self._annotation_files)

    def __getitem__(self, index: int) -> dict:
        import torch  # deferred: parsing-only users of this module need no torch

        annotation_file = self._annotation_files[index]
        stem = annotation_file.stem
        image_path = self._image_dir / f"{stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"annotation {stem}.txt has no image at {image_path}")

        crops = parse_annotation_text(annotation_file.read_text())
        with Image.open(image_path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)  # HWC -> CHW
        boxes = torch.tensor([(c.x1, c.y1, c.x2, c.y2) for c in crops], dtype=torch.float32)
        scores = torch.tensor([c.score for c in crops], dtype=torch.float32)
        sample = {
            "image": image,
            "boxes": boxes.reshape(-1, 4),
            "scores": scores,
            "image_id": stem,
        }
        # Sample-level, not image-level: geometric transforms (resize, flip)
        # must move the boxes together with the pixels or the labels lie.
        if self._transform is not None:
            sample = self._transform(sample)
        return sample
