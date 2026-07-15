"""Contract for the HTTP layer.

A fake pipeline is injected through the app factory, so these tests cover
request parsing, validation, serialization and error mapping - not ML.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from recompose.pipeline import AnalysisResult
from recompose.types import Box, CropCandidate, ScoredCrop


class FakePipeline:
    def __init__(self):
        self.calls = []

    def analyze(self, image, aspects, top_k):
        height, width = image.shape[:2]
        self.calls.append({"shape": image.shape, "aspects": tuple(aspects), "top_k": top_k})
        scored = ScoredCrop(
            crop=CropCandidate(x=0, y=0, w=width // 2, h=height // 2, aspect="1:1"),
            score=0.8,
            rule_scores={"thirds": 0.9, "retention": 0.7},
        )
        return AnalysisResult(
            width=width,
            height=height,
            subjects=[Box(0, 0, width / 4, height / 4, label="dog", confidence=0.9)],
            horizon_tilt_deg=-2.0,
            saliency=np.zeros((height, width), dtype=np.float32),
            crops_by_aspect={aspect: [scored] for aspect in aspects},
            constraints_relaxed=False,
        )


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline):
    with TestClient(create_app(lambda: fake_pipeline)) as test_client:
        yield test_client


def png_upload(width=64, height=48):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, "PNG")
    return {"file": ("photo.png", buffer.getvalue(), "image/png")}


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyze:
    def test_happy_path_returns_normalized_geometry(self, client):
        response = client.post("/api/analyze?aspects=original,1:1", files=png_upload())
        assert response.status_code == 200
        body = response.json()
        assert body["image"] == {"width": 64, "height": 48}
        assert body["horizon_tilt_deg"] == pytest.approx(-2.0)
        assert body["constraints_relaxed"] is False
        assert set(body["crops"]) == {"original", "1:1"}
        crop = body["crops"]["1:1"][0]
        for key in ("x", "y", "w", "h"):
            assert 0.0 <= crop[key] <= 1.0
        assert crop["w"] == pytest.approx(0.5)
        assert crop["rules"]["thirds"] == pytest.approx(0.9)
        subject = body["subjects"][0]
        assert subject["label"] == "dog"
        assert 0.0 <= subject["x2"] <= 1.0

    def test_saliency_returned_as_png_data_uri(self, client):
        response = client.post("/api/analyze", files=png_upload())
        assert response.json()["saliency_png"].startswith("data:image/png;base64,")

    def test_request_parameters_forwarded_to_pipeline(self, client, fake_pipeline):
        client.post("/api/analyze?aspects=4:5&top_k=5", files=png_upload())
        assert fake_pipeline.calls[-1]["aspects"] == ("4:5",)
        assert fake_pipeline.calls[-1]["top_k"] == 5

    def test_default_aspects_cover_all_supported(self, client, fake_pipeline):
        client.post("/api/analyze", files=png_upload())
        assert fake_pipeline.calls[-1]["aspects"] == ("original", "1:1", "4:5", "16:9")

    def test_oversized_image_downscaled_before_analysis(self, client, fake_pipeline):
        client.post("/api/analyze", files=png_upload(width=2048, height=1024))
        height, width, _ = fake_pipeline.calls[-1]["shape"]
        assert max(width, height) == 1024
        assert width / height == pytest.approx(2.0, rel=0.01)

    def test_non_image_upload_rejected(self, client):
        files = {"file": ("notes.txt", b"not an image", "text/plain")}
        assert client.post("/api/analyze", files=files).status_code == 400

    def test_malformed_aspect_rejected(self, client):
        response = client.post("/api/analyze?aspects=4x5", files=png_upload())
        assert response.status_code == 400

    def test_out_of_range_top_k_rejected(self, client):
        assert client.post("/api/analyze?top_k=0", files=png_upload()).status_code == 422
