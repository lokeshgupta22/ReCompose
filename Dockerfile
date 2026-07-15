# Two stages: build the static frontend with Node, then run everything from
# one Python container that serves both the API and the built React app on
# a single port - what a HF Spaces "Docker" SDK deployment expects.

FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS backend

# opencv/numpy/onnxruntime wheels sometimes link against these at import
# time even though we use the "headless" opencv build; installing them
# up front avoids a failed build discovering it mid-way through model load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HF Spaces containers run as a non-root user by convention; matching that
# here (rather than relying on root) avoids permission errors on any path
# their infra marks read-only for root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    RECOMPOSE_CACHE_DIR=/home/user/.cache/recompose
WORKDIR /home/user/app

# Dependencies before source: this layer only invalidates when
# pyproject.toml/uv.lock change, not on every code edit.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY --chown=user recompose/ recompose/
COPY --chown=user api/ api/
COPY --from=frontend --chown=user /app/web/dist web/dist

# Bake model weights into the image at build time. Without this, the
# container's first request (or worse, its first request after waking
# from an idle sleep) would block on downloading ~180MB from GitHub/PyTorch
# Hub - slow, and a single point of failure if that host is unreachable
# from the runtime sandbox.
RUN uv run python -c "\
from recompose.perception.saliency import SaliencyEstimator; \
from recompose.perception.detection import SubjectDetector; \
SaliencyEstimator(); \
SubjectDetector()"

EXPOSE 7860
CMD ["uv", "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
