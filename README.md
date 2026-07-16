# ReCompose

**Upload a photo. Get it professionally composed — and see exactly why.**

**Live demo: [recompose.onrender.com](https://recompose.onrender.com/)** — runs on a
free 512MB / 0.1-CPU instance, so expect ~1 min to wake from idle sleep and a few
seconds per photo. (How the whole pipeline fits in 512MB is a design story of its
own — see the perception notes below.)

ReCompose finds the subject of a photograph, understands its composition (rule of
thirds, balance, headroom, horizon), and returns the best crop for every target
aspect ratio, each with a per-rule score breakdown explaining the choice. Every
photographer gets a composition assistant; nobody has to memorize the rules.

- **v1 (this release)**: complete rule-based pipeline — pretrained perception models,
  grid-anchor candidate generation, hard safety constraints, five-rule composition
  scorer, FastAPI backend, React frontend. ~3 s per photo on CPU.
- **Roadmap**: a learned crop scorer trained on the GAICD benchmark and a NIMA-style
  aesthetic model trained on AVA, benchmarked against this baseline (see
  [Roadmap](#roadmap)).

## Why not just ask a vision LLM?

A VLM can critique a photo, but it can't reliably return precise, deterministic crop
geometry; it costs per call, adds seconds of network latency, and your photos leave
the machine. ReCompose is a purpose-built pipeline: deterministic, explainable
per-rule, fast enough for CPU, and fully local. The design also cleanly separates
*safety* (hard constraints that never cut a subject) from *taste* (a swappable
scoring model) — something a single opaque model call can't offer.

## How it works

```
photo ──► 1. Perception ──► 2. Candidate crops ──► 3. Constraints ──► 4. Scoring ──► best crops + why
```

### 1. Perception — three "senses"

| Sense | Model | Output | Role downstream |
|---|---|---|---|
| Saliency | U²-Net-p (pretrained, ONNX Runtime; full U²-Net via `RECOMPOSE_SALIENCY_MODEL=u2net`) | H×W map in [0,1] of where the eye is drawn | Soft signal: drives scoring |
| Subjects | YOLOv8n exported to ONNX, filtered to photographic subject classes | labeled boxes | Hard constraint: never cut a subject |
| Horizon | Classical CV (Canny → HoughLinesP) | tilt angle or `None` | Straightening suggestion |

Design notes:

- **U²-Net runs directly on `onnxruntime`** — no wrapper library. We originally used
  `rembg`, but its transitive chain (`pymatting → numba → llvmlite`) broke dependency
  resolution, and we only ever used it as a 40-line convenience around the same ONNX
  session. Owning the preprocessing (resize to 320², ImageNet mean/std, HWC→CHW) also
  means we control exactly what the model sees.
- **YOLOv8n also runs on `onnxruntime`, not ultralytics.** Ultralytics transitively
  installs CUDA-enabled torch — hundreds of MB resident, which OOMed the 512MB
  deploy target. It's now export-time tooling only (`uv sync --extra export`); the
  runtime ships a 12MB `.onnx` plus hand-rolled letterbox/decode/NMS, all unit-tested
  against synthetic tensors. Sessions run arena-off and single-threaded: measured
  421MB peak / 200MB steady for the whole server, inside the 512MB free-tier limit.
- **U²-Net-p (4.4MB) is the default saliency model**: full U²-Net's first inference
  peaks over 512MB on its own. On the reference photo, 3 of 4 aspect ratios choose
  identical crops under either model — boxes do the structural work, saliency ranks.
- **Saliency stays continuous** (never thresholded to a binary mask): the weighted
  centroid must let a bright face pull harder than a dim jacket, and retention scoring
  must price losing a face higher than losing a shadow. Quantize a signal only when
  forced to — every downstream consumer inherits the information loss.
- **Boxes are hard constraints, saliency is a soft signal.** Saliency alone can pass a
  crop through a person's knees (85% of the glow survives); boxes alone are blind to
  anything outside 80 COCO classes and can't grade quality. Guardrail + ranker.
- **Horizon uses a length-weighted median** of near-horizontal line angles: robust to
  clutter (a tilted dune edge can't drag the estimate the way a mean would), and it
  honestly returns `None` for scenes with no horizontal structure instead of
  hallucinating an angle.

### 2. Candidate generation — grid anchors (Zeng et al., CVPR 2019)

A 4000×3000 photo contains ~10¹³ possible crop rectangles, but aesthetic quality
varies *smoothly* with crop coordinates — nobody can see a 3-pixel reframe. So we
sample the space coarsely instead of exhaustively: **5 scales** (55–100% of the
largest crop that fits the target ratio) × **5×5 anchor positions** × **4 aspect
ratios** (`original`, `1:1`, `4:5`, `16:9` — chosen for where photos actually get
posted) ≈ 500 candidates, deduplicated. Full coverage of meaningfully-different
framings, near-zero redundancy. Pure geometry, zero ML imports, fully unit-tested.

This is the same discretization the GAIC paper uses, which matters for the roadmap:
the future learned scorer ranks *exactly these candidates*, scoring all of them in
one forward pass (shared backbone feature map + per-candidate RoIAlign — the Fast
R-CNN trick). The generator never changes; only the judge gets swapped.

### 3. Hard constraints — pass/fail, then a relaxation ladder

- **Subject integrity**: a crop may not slice through any detected subject box, and
  must keep the primary (largest) subject. Secondary subjects may be *cleanly
  excluded* — dropping a background passer-by is legitimate recomposition; cutting
  them in half is not. Containment uses tolerance bands (≥95% = contained,
  ≤5% = excluded) because detector boxes are noisy measurements: never compare
  noisy values with `==`.
- **Saliency retention**: candidates must keep ≥55% of total visual mass.
- **Relaxation ladder**: if strict filtering leaves nothing (e.g. a 16:9 crop of a
  portrait-orientation photo *cannot* retain 55%), constraints relax in value order —
  retention is sacrificed before subject integrity, because a loose crop is a taste
  problem while an amputated subject is a defect. The response carries a
  `constraints_relaxed` flag so the UI can be honest about it. Fallbacks trigger on
  *observed infeasibility*, never on hand-enumerated special cases.

Constraints are strategy objects behind a single-method `Protocol`; the filter
composes them (open/closed — a new rule is a new class, not a modified filter).

### 4. Scoring — five rules, weighted fusion

Every rule maps a crop to [0,1], or `None` when it doesn't apply:

| Rule | Weight | Definition |
|---|---|---|
| Rule of thirds | 0.40 | `exp(−(d/σ)²)` where `d` = distance from the saliency centroid to the nearest thirds power point (crop-relative), σ = 1/6 (one grid cell) |
| Retention | 0.25 | fraction of total visual mass kept (soft twin of the hard constraint: the gate defines *valid*, the score prefers *better among valid*) |
| Balance | 0.15 | `1 − |L−R| / (L+R)` over left/right half mass |
| Headroom | 0.10 | trapezoid over the gap above the topmost subject: full marks 4–18% of crop height |
| Subject size | 0.10 | trapezoid over the primary subject's share of crop area: full marks 8–50% |

Details that matter:

- **The centroid is O(1) per crop** via summed-area tables of the saliency map and
  its x/y moments (build once per photo, four lookups per query) — the Viola–Jones
  integral-image idea. Filtering and scoring hundreds of candidates costs
  microseconds.
- **`None` ≠ 0.** Inapplicable rules (headroom on a landscape with no subjects) are
  skipped *and their weight leaves the denominator*, so a ramen photo is judged only
  by rules that have an opinion about ramen. "No opinion" and "strong negative
  opinion" must never share a value.
- **Trapezoid plateaus encode honest uncertainty** — claiming the ideal headroom is
  exactly 10% would be false precision.
- **The weights are hand-set on purpose.** This scorer is the measured baseline the
  learned model must beat; hand-tuned-heuristics vs learned-scoring is the headline
  ablation of Phase 2.

## API

`POST /api/analyze` (multipart image upload; `aspects` and `top_k` query params) →

```jsonc
{
  "image": {"width": 768, "height": 1024},
  "horizon_tilt_deg": 3.05,
  "constraints_relaxed": true,
  "subjects": [{"x1": 0.06, "y1": 0.38, "x2": 0.31, "y2": 0.85, "label": "person", "confidence": 0.87}],
  "saliency_png": "data:image/png;base64,…",
  "crops": {
    "4:5": [{"x": 0.06, "y": 0.21, "w": 0.77, "h": 0.72, "score": 0.734,
             "rules": {"thirds": 0.53, "retention": 0.83, "balance": 0.88,
                        "headroom": 0.81, "subject_size": 1.0}}]
  }
}
```

Decisions:

- **All geometry is normalized to [0,1]** fractions of the analyzed image. The server
  analyzes a bounded copy, the browser renders at layout size, the original may be
  8000px wide — fractions make every consumer resolution-independent and kill the
  entire class of "whose pixels?" overlay bugs.
- **EXIF orientation is applied before analysis** — phone cameras store rotation as
  metadata, and skipping this step silently analyzes portrait shots lying sideways.
- **Analysis runs at ≤1024px**: composition is scale-invariant, the perception models
  downsample internally anyway, and the summed-area tables are O(pixels) in memory.
- **Models load once at startup** (FastAPI lifespan) so failures surface at deploy
  time, not on a user's request. The app is built by a factory that accepts a
  pipeline provider, so API tests inject a fake and never touch model weights;
  `api/server.py` is the only module that constructs the real thing.

## Frontend

React + Vite, no framework beyond that. Upload (drag-and-drop), aspect-ratio tabs,
top-3 candidate chips, before/after views, toggleable overlays (subject boxes,
saliency heatmap, crop outline), a thirds grid on the result, per-rule score bars,
and a lossless PNG download rendered from the *original* resolution image. All
overlay geometry is the API's normalized fractions applied as CSS percentages — the
client does no pixel math at all.

## Engineering practices

- **TDD throughout**: 134 tests, each written before its implementation (red → green).
  Geometry, constraints, scoring, and the pipeline are all testable without any model
  download — perception models sit behind `Protocol`s and tests inject fakes
  (dependency inversion).
- **The e2e run earned its keep immediately**: it caught a real bug unit tests
  couldn't — OpenCV changed `HoughLinesP`'s output shape between versions
  (`(N,1,4)` → `(N,4)`), crashing horizon estimation. Fixed with a shape-normalizing
  `reshape(-1, 4)` plus regression tests that run the real OpenCV path against
  synthetic scenes with known tilt. Unit tests verify your logic; integration tests
  verify your assumptions about everyone else's.
- **Reproducible environment**: `uv` with a lockfile and a pinned Python 3.12
  (`.python-version`) after the resolver, left unpinned, picked Python 3.14 and
  backtracked into a 2021 `llvmlite` that couldn't build.
- **Lint**: ruff (`E,F,I,W,UP,B`), zero suppressions — findings get fixed, not
  silenced.
- Small, single-purpose commits; the git log reads as the project's build diary.

## Project structure

```
recompose/            core library (no web concerns)
├── perception/       saliency (U²-Net/ONNX), detection (YOLOv8n), horizon (OpenCV)
├── cropping/         grid-anchor candidate generator, constraint strategy objects
├── scoring/          five composition rules + weighted fusion scorer
├── saliency_stats.py summed-area tables: O(1) region mass and centroid
├── pipeline.py       orchestration + relaxation ladder (models injected)
└── types.py          Box, CropCandidate, ScoredCrop — dependency-free geometry
api/                  FastAPI: schemas, imaging helpers, app factory, uvicorn entry
web/                  React + Vite client
tests/                134 tests; run without downloading any model
```

## Getting started

```bash
# backend (Python 3.12 via uv; first run downloads U²-Net-p ~4MB and exports YOLOv8n
# to ONNX - the export needs the dev/export extras, hence --all-extras)
uv sync --all-extras
uv run pytest                                   # 134 tests, no downloads needed
uv run uvicorn api.server:app --port 8000

# frontend
cd web && npm install && npm run dev            # http://localhost:5173, proxies /api
```

Model weights cache under `~/.cache/recompose` (override: `RECOMPOSE_CACHE_DIR`).

## Roadmap

- **Phase 2 — learned crop scorer.** Train on GAICD (1,236 images × ~86 human-rated
  candidate crops): shared CNN backbone + RoIAlign over candidates + scoring head.
  Evaluate with SRCC and Acc@K against published baselines; ablate backbone size and
  a saliency input channel; report learned-vs-rule-based as the headline comparison.
- **Phase 3 — aesthetic model.** NIMA-style score-distribution head trained with EMD
  loss on an AVA subset; fuse with the crop scorer; show before/after aesthetic
  deltas in the UI.
- **Phase 4 — ship.** ONNX export + INT8 quantization with a latency table, warmup
  inference before readiness, Docker, CI, and a public demo on Hugging Face Spaces.

## Status

v1 — rule-based baseline complete and verified end-to-end (M-series MacBook, CPU
only: ~3 s per photo including all three perception models).
