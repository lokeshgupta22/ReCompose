"""Response models. All geometry is normalized to [0, 1] fractions of the
analyzed image so clients can overlay results at any display resolution."""

from __future__ import annotations

from pydantic import BaseModel


class ImageInfo(BaseModel):
    width: int
    height: int


class SubjectOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float


class CropOut(BaseModel):
    x: float
    y: float
    w: float
    h: float
    score: float
    rules: dict[str, float]


class AnalyzeResponse(BaseModel):
    image: ImageInfo
    horizon_tilt_deg: float | None
    constraints_relaxed: bool
    subjects: list[SubjectOut]
    saliency_png: str
    crops: dict[str, list[CropOut]]
