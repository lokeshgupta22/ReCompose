# ReCompose — AI Photo Recomposition (Smart Re-Crop)

## Context

Goal: an app where a photographer uploads a photo and gets back a professionally composed version — the system finds the subject, understands composition (rule of thirds, headroom, horizon, balance), and returns the best crop(s) with a visual explanation of *why*.

Equally important goal: this must be a standout **AI/ML engineer resume project**. That means the ML work is real (trained/fine-tuned models, public-benchmark numbers, ablations, an eval harness) rather than API glue, and the result is a deployable product with MLOps polish (ONNX export, Docker, CI, live demo).

Decisions already made with the user:
- **Fix method**: smart re-crop (not generative outpainting, not critique-only)
- **ML depth**: train/fine-tune own models
- **Platform**: web app (FastAPI backend + React frontend)
- **Timeline**: ~1–2 months, phased so there's a demoable product early

## Architecture Overview

Three-stage inference pipeline:

```
photo → [1. Perception] → [2. Candidate crops] → [3. Scoring & ranking] → best crop + explanation
```

1. **Perception** — pretrained models (not trained by us; that's fine, this is plumbing):
   - Saliency map: U²-Net (or ISNet) → where the visual subject mass is
   - Subject/face detection: YOLOv8n (person/face) → hard constraints so crops never amputate a subject
   - Horizon estimation: OpenCV Hough lines on edge map → tilt angle for auto-straightening
2. **Candidate crop generation** — grid-anchor method (from the GAIC paper): generate ~100–500 candidate crops across aspect ratios (original, 1:1, 4:5, 16:9), filter out any that cut a detected subject box or drop >X% of saliency mass.
3. **Scoring & ranking** — this is the trained-model core (see below). Final score = learned crop score, with interpretable rule-based features (thirds distance, headroom, balance) computed alongside for the explanation overlay.

Output: top crop per aspect ratio, plus overlays — thirds grid, saliency heatmap, subject boxes, per-rule scores ("subject 0.92 aligned to right-third power point; horizon leveled by 2.3°").

## The ML Work (the resume core)

Two trained models, in order of priority:

### Model A — Crop scorer trained on GAICD (weeks 3–4)
- **Dataset**: GAICD (Grid Anchor based Image Cropping Database): ~1,200 train / ~200 test images, each with ~90 pre-defined candidate crops human-rated 1–5. Small enough to train on a free Colab/Kaggle GPU or Apple Silicon.
- **Model**: timm backbone (start MobileNetV3 or EfficientNet-B0, ablate up to ViT-S) + RoIAlign over candidate crop regions → shared MLP head scoring all candidates in one forward pass (this is the GAIC architecture — efficient and well documented).
- **Metrics** (standard for this benchmark, published baselines exist to compare against): SRCC (rank correlation with human ratings), Acc@1 and Acc@5 (does a top-K crop match a human-preferred crop). Report a table vs. GAIC / CGS / TransView published numbers.
- **Resume ablations**: backbone size vs. accuracy/latency; with vs. without saliency channel as a 4th input channel; learned score vs. pure rule-based baseline from Phase 1.

### Model B — Aesthetic quality model, NIMA-style on AVA (weeks 5–6, stretch but high-value)
- **Dataset**: AVA — 255k images with 1–10 score histograms. Full dataset is ~32 GB; use a curated ~30–50k subset (script the sampling, stratified by score) to keep it tractable.
- **Model**: timm backbone + 10-way softmax head, trained with **EMD (Earth Mover's Distance) loss** on the score distribution — the NIMA recipe. Metrics: SRCC/PLCC vs. mean human score.
- **Role in pipeline**: global "is this a good photo" signal fused with Model A's crop score; also powers a before/after aesthetic score delta shown in the UI ("6.1 → 7.4").

### Training infrastructure (what interviewers actually probe)
- PyTorch + timm, mixed precision, cosine LR schedule, early stopping
- **Experiment tracking**: Weights & Biases (free tier) — every run logged, config-driven via YAML/Hydra
- **Eval harness**: `recompose eval` CLI that reproduces every number in the README from a checkpoint — one command, deterministic
- Seed control + config snapshots so results are reproducible

## Product & MLOps Layer

- **Backend**: FastAPI — `POST /analyze` (upload → crops + overlays + scores as JSON), model loaded once at startup; export models to **ONNX**, serve with onnxruntime (report PyTorch vs. ONNX vs. INT8-quantized latency table — easy, impressive MLOps content)
- **Frontend**: React + Vite (simple, no Next.js needed): upload dropzone, before/after slider, aspect-ratio tabs, toggleable overlays (thirds grid / saliency heatmap / subject boxes), per-rule score breakdown panel
- **Packaging**: Dockerfile (multi-stage, slim runtime image), docker-compose for local dev
- **Deploy**: Hugging Face Spaces (free, GPU-optional since ONNX-CPU inference is fast at this model size) — gives a permanent public demo link for the resume
- **CI**: GitHub Actions — ruff + pytest on push; tests cover crop generation geometry, constraint filtering, and API contract
- **README as the deliverable**: problem statement, architecture diagram, metrics table vs. published baselines, ablation table, latency table, demo GIF, link to live demo. Written like a mini paper.

## Repo Structure

```
ReCompose/
├── recompose/            # Python package
│   ├── perception/       # saliency, detection, horizon
│   ├── cropping/         # candidate generation, constraints
│   ├── scoring/          # Model A, Model B, rule-based features
│   ├── pipeline.py       # end-to-end inference
│   └── eval/             # benchmark harness (GAICD, AVA)
├── training/             # train scripts + Hydra configs
├── api/                  # FastAPI app
├── web/                  # React frontend
├── tests/
├── notebooks/            # EDA, ablation analysis
├── Dockerfile / docker-compose.yml
└── README.md
```

## Phased Milestones (~8 weeks)

**Phase 1 (wk 1–2) — Working baseline, no training yet**
Repo scaffold, pretrained U²-Net + YOLOv8 perception, grid-anchor candidate generation, *rule-based* scoring (thirds distance, saliency balance, headroom, horizon), FastAPI + minimal React UI. End-to-end demo works. This baseline is also the control for later ablations.

**Phase 2 (wk 3–4) — Model A (GAICD crop scorer)**
Dataset download/loader, training loop with W&B, eval harness with SRCC/Acc@K, integrate behind a config flag, beat the rule-based baseline and record both numbers.

**Phase 3 (wk 5–6) — Model B (NIMA on AVA subset) + fusion**
AVA subset script, EMD-loss training, score fusion with Model A, before/after aesthetic delta in UI, ablation tables.

**Phase 4 (wk 7–8) — Ship it**
ONNX export + quantization + latency table, Docker, HF Spaces deploy, CI, README with all metrics + demo GIF. Optional: short blog post walking through the ablations.

Fallback if time runs short: Phases 1–2 + Phase 4 alone is already a complete, benchmarked, deployed project; Phase 3 is the stretch layer.

## Verification

- **Per phase**: run the pipeline on a fixed set of ~10 test photos (portrait, landscape, off-center subject, tilted horizon) and eyeball crops + overlays in the UI
- **Model A**: `recompose eval --benchmark gaicd` reproduces SRCC/Acc@K on the official test split; sanity threshold — must beat the rule-based baseline
- **Model B**: SRCC/PLCC on AVA held-out split vs. NIMA published numbers (expect somewhat lower on a subset — document that honestly)
- **API**: pytest hitting `/analyze` with sample images, asserting response schema and crop-geometry invariants (crops within bounds, subject boxes uncut)
- **Deploy**: upload a photo on the live HF Space and confirm end-to-end latency < ~2s CPU

## First Implementation Steps (when execution starts)

1. `git init`, Python project scaffold (uv or pip-tools), package skeleton per repo structure
2. Perception module with pretrained U²-Net + YOLOv8n, visualized in a notebook
3. Grid-anchor candidate generator + constraint filter with unit tests
4. Rule-based scorer + `pipeline.py` end-to-end
5. FastAPI `/analyze` + React upload/before-after UI
