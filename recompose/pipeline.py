"""End-to-end analysis: perception -> candidates -> constraints -> scoring.

Perception models are injected behind Protocols (dependency inversion), so
the pipeline is testable with fakes and the heavyweight defaults are only
imported when explicitly requested via `with_default_models`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recompose.cropping import (
    CandidateGridConfig,
    SaliencyRetentionConstraint,
    SubjectIntegrityConstraint,
    filter_candidates,
    generate_candidates,
)
from recompose.perception.horizon import estimate_horizon_tilt
from recompose.saliency_stats import SaliencyStats
from recompose.scoring import CompositionScorer, build_default_scorer
from recompose.types import Box, ScoredCrop

DEFAULT_ASPECTS = ("original", "1:1", "4:5", "16:9")
DEFAULT_MIN_RETENTION = 0.55


class SaliencyModel(Protocol):
    def predict(self, image: np.ndarray) -> np.ndarray: ...


class SubjectModel(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...


@dataclass
class AnalysisResult:
    width: int
    height: int
    subjects: list[Box]
    horizon_tilt_deg: float | None
    saliency: np.ndarray
    crops_by_aspect: dict[str, list[ScoredCrop]]
    constraints_relaxed: bool


class RecomposePipeline:
    def __init__(
        self,
        saliency_model: SaliencyModel,
        subject_model: SubjectModel,
        horizon_fn: Callable[[np.ndarray], float | None] = estimate_horizon_tilt,
        grid_config: CandidateGridConfig | None = None,
        min_retention: float = DEFAULT_MIN_RETENTION,
        scorer_factory: Callable[
            [SaliencyStats, Sequence[Box]], CompositionScorer
        ] = build_default_scorer,
    ):
        self._saliency_model = saliency_model
        self._subject_model = subject_model
        self._horizon_fn = horizon_fn
        self._grid_config = grid_config
        self._min_retention = min_retention
        self._scorer_factory = scorer_factory

    @classmethod
    def with_default_models(cls, **kwargs) -> RecomposePipeline:
        """Build with the pretrained U²-Net + YOLOv8n models (downloads weights
        on first use)."""
        from recompose.perception.detection import SubjectDetector
        from recompose.perception.saliency import SaliencyEstimator

        return cls(SaliencyEstimator(), SubjectDetector(), **kwargs)

    def analyze(
        self,
        image: np.ndarray,
        aspects: Sequence[str] = DEFAULT_ASPECTS,
        top_k: int = 3,
    ) -> AnalysisResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got shape {image.shape}")
        height, width = image.shape[:2]

        saliency = self._saliency_model.predict(image)
        if saliency.shape != (height, width):
            raise ValueError(
                f"saliency shape {saliency.shape} does not match image {(height, width)}"
            )
        subjects = self._subject_model.detect(image)
        stats = SaliencyStats(saliency)
        scorer = self._scorer_factory(stats, subjects)

        crops_by_aspect: dict[str, list[ScoredCrop]] = {}
        any_relaxed = False
        for aspect in aspects:
            candidates = generate_candidates(width, height, aspect, self._grid_config)
            survivors, relaxed = self._apply_constraints(candidates, subjects, stats)
            any_relaxed |= relaxed
            scored = sorted(
                (scorer.score(crop) for crop in survivors),
                key=lambda s: s.score,
                reverse=True,
            )
            crops_by_aspect[aspect] = scored[:top_k]

        return AnalysisResult(
            width=width,
            height=height,
            subjects=subjects,
            horizon_tilt_deg=self._horizon_fn(image),
            saliency=saliency,
            crops_by_aspect=crops_by_aspect,
            constraints_relaxed=any_relaxed,
        )

    def _apply_constraints(self, candidates, subjects, stats):
        """Strict first; relax rather than return nothing.

        Ladder: subject integrity + retention -> subject integrity only ->
        unfiltered. Never sacrifice subject integrity before giving up
        entirely: a low-retention crop is a taste problem, an amputated
        subject is a defect.
        """
        strict = [
            SubjectIntegrityConstraint(subjects),
            SaliencyRetentionConstraint(stats, self._min_retention),
        ]
        survivors = filter_candidates(candidates, strict)
        if survivors:
            return survivors, False
        survivors = filter_candidates(candidates, [SubjectIntegrityConstraint(subjects)])
        if survivors:
            return survivors, True
        return list(candidates), True
