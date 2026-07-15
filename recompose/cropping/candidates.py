"""Grid-anchor candidate crop generation (Zeng et al., CVPR 2019).

Aesthetic quality varies smoothly with crop coordinates, so instead of the
~10^13 possible rectangles we sample a coarse grid: a few sizes x a few
positions x the requested aspect ratio. Pure geometry - no ML imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from recompose.types import CropCandidate

DEFAULT_SCALES = (0.55, 0.66, 0.77, 0.88, 1.0)


@dataclass(frozen=True)
class CandidateGridConfig:
    """Knobs of the grid: crop sizes (as fractions of the largest crop that
    fits the target ratio) and how many anchor positions per axis."""

    scales: tuple[float, ...] = DEFAULT_SCALES
    positions_per_axis: int = 5


def parse_aspect(aspect: str) -> float | None:
    """'original' -> None (keep image ratio); 'W:H' -> W/H. Raises on nonsense."""
    if aspect == "original":
        return None
    parts = aspect.split(":")
    if len(parts) != 2:
        raise ValueError(f"aspect must be 'original' or 'W:H', got {aspect!r}")
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"aspect must be numeric 'W:H', got {aspect!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"aspect sides must be positive, got {aspect!r}")
    return w / h


def generate_candidates(
    width: int,
    height: int,
    aspect: str = "original",
    config: CandidateGridConfig | None = None,
) -> list[CropCandidate]:
    """Generate deduplicated crop windows of the given aspect inside the image."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    config = config or CandidateGridConfig()

    ratio = parse_aspect(aspect) or width / height
    max_w = min(float(width), height * ratio)
    max_h = max_w / ratio

    seen: set[CropCandidate] = set()
    candidates: list[CropCandidate] = []
    for scale in config.scales:
        w = round(max_w * scale)
        h = round(max_h * scale)
        if w <= 0 or h <= 0:
            continue
        for x in _anchor_offsets(width - w, config.positions_per_axis):
            for y in _anchor_offsets(height - h, config.positions_per_axis):
                candidate = CropCandidate(x=x, y=y, w=w, h=h, aspect=aspect)
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
    return candidates


def _anchor_offsets(free_space: int, positions: int) -> list[int]:
    """Evenly spread anchor offsets over the room the window can slide in."""
    if free_space <= 0 or positions <= 1:
        return [0]
    return sorted({round(free_space * i / (positions - 1)) for i in range(positions)})
