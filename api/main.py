"""HTTP layer: thin routes over the analysis pipeline.

The pipeline is supplied by a factory so tests inject fakes; the default
factory loads the real pretrained models once at startup (lifespan), never
per request.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.imaging import decode_image, downscale, saliency_data_uri
from api.schemas import AnalyzeResponse, CropOut, ImageInfo, SubjectOut
from recompose.cropping import parse_aspect
from recompose.pipeline import DEFAULT_ASPECTS, AnalysisResult, RecomposePipeline
from recompose.types import Box, ScoredCrop


class AnalysisPipeline(Protocol):
    def analyze(
        self, image: np.ndarray, aspects: tuple[str, ...], top_k: int
    ) -> AnalysisResult: ...


def create_app(
    pipeline_factory: Callable[[], AnalysisPipeline] = RecomposePipeline.with_default_models,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="ReCompose API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # dev default; restrict per deployment
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(
        request: Request,
        file: Annotated[UploadFile, File()],
        aspects: Annotated[str, Query()] = ",".join(DEFAULT_ASPECTS),
        top_k: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> AnalyzeResponse:
        try:
            image = decode_image(await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        aspect_list = tuple(a.strip() for a in aspects.split(",") if a.strip())
        try:
            for aspect in aspect_list:
                parse_aspect(aspect)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = request.app.state.pipeline.analyze(
            downscale(image), aspects=aspect_list, top_k=top_k
        )
        return _serialize(result)

    return app


def _serialize(result: AnalysisResult) -> AnalyzeResponse:
    return AnalyzeResponse(
        image=ImageInfo(width=result.width, height=result.height),
        horizon_tilt_deg=result.horizon_tilt_deg,
        constraints_relaxed=result.constraints_relaxed,
        subjects=[_subject_out(box, result.width, result.height) for box in result.subjects],
        saliency_png=saliency_data_uri(result.saliency),
        crops={
            aspect: [_crop_out(scored, result.width, result.height) for scored in scored_list]
            for aspect, scored_list in result.crops_by_aspect.items()
        },
    )


def _subject_out(box: Box, width: int, height: int) -> SubjectOut:
    return SubjectOut(
        x1=box.x1 / width,
        y1=box.y1 / height,
        x2=box.x2 / width,
        y2=box.y2 / height,
        label=box.label,
        confidence=box.confidence,
    )


def _crop_out(scored: ScoredCrop, width: int, height: int) -> CropOut:
    crop = scored.crop
    return CropOut(
        x=crop.x / width,
        y=crop.y / height,
        w=crop.w / width,
        h=crop.h / height,
        score=scored.score,
        rules=scored.rule_scores,
    )
