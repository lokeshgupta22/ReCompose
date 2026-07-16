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

from dataclasses import dataclass


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
