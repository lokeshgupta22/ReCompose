# Three stages: Node builds the static frontend, a throwaway Python stage
# exports/downloads model weights (this is the only place torch ever
# exists), and a slim runtime serves API + frontend from one process/port.
# Target: Render free tier - 512MB RAM is the constraint that shaped this.

FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

# Weights stage: turns yolov8n.pt into yolov8n.onnx and fetches u2net.onnx.
# ultralytics drags in CUDA-enabled torch (hundreds of MB resident) - fine
# here because nothing from this stage ships except two .onnx files.
FROM python:3.12-slim AS weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /weights

# Install from the same lockfile the app uses so the exporting ultralytics
# version is exactly the one pinned in uv.lock, not whatever pip resolves
# the day the image happens to rebuild.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra export

RUN uv run python -c "\
from ultralytics import YOLO; \
YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
RUN uv run python -c "\
import urllib.request; \
urllib.request.urlretrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx', 'u2netp.onnx')"

FROM python:3.12-slim AS backend

# opencv/numpy/onnxruntime wheels sometimes link against these at import
# time even though we use the "headless" opencv build; installing them
# up front avoids a failed build discovering it mid-way through model load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Run as non-root: standard container hygiene, and some hosts mark paths
# read-only for root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    RECOMPOSE_CACHE_DIR=/home/user/.cache/recompose
WORKDIR /home/user/app

# Dependencies before source: this layer only invalidates when
# pyproject.toml/uv.lock change, not on every code edit. --no-install-project
# installs only third-party dependencies and skips building our own
# `recompose` package - which needs README.md and the source tree, neither
# copied yet, and would otherwise fail hatchling's metadata validation here.
# No --extra export here: the runtime stays torch-free by construction.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY --chown=user recompose/ recompose/
COPY --chown=user api/ api/
COPY --from=frontend --chown=user /app/web/dist web/dist

# Now install the local project itself. Dependencies are already cached
# from the layer above, so this is fast regardless of how often source
# changes - only this final sync re-runs on every code edit.
RUN uv sync --frozen

# Bake weights into the image where the app's cache lookup expects them.
# Without this, the container's first request after an idle-sleep wake
# would block on a download - slow, and a single point of failure.
COPY --from=weights --chown=user /weights/yolov8n.onnx /weights/u2netp.onnx \
     /home/user/.cache/recompose/

# Smoke test: both InferenceSessions must construct from the baked weights.
# A corrupt or missing file fails the build here, not a user's request.
RUN uv run python -c "\
from recompose.perception.saliency import SaliencyEstimator; \
from recompose.perception.detection import SubjectDetector; \
SaliencyEstimator(); \
SubjectDetector()"

# Render injects PORT; default to 7860 for local runs. Shell form so the
# env var actually expands - exec-form CMD arrays don't go through a shell.
EXPOSE 7860
CMD ["sh", "-c", "uv run uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
