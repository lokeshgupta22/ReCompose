# ReCompose — handoff

This file exists so anyone (a person, or a fresh AI assistant with no memory of this
conversation) can pick up exactly where we left off, in one prompt. It captures not
just *what* was built but *how we work together* and *why* every non-obvious decision
was made. Read this before touching code.

Repo: [github.com/lokeshgupta22/ReCompose](https://github.com/lokeshgupta22/ReCompose)
(branch `main`, currently at commit `92414d6`, working tree clean, no other branches).

---

## 1. What this project is and why it exists

ReCompose is a photo-composition app: upload a photo, get back a professionally
cropped version (rule of thirds, balance, headroom, horizon) with a per-rule score
explaining *why* that crop is good.

The explicit, stated purpose is **not just a working app** — it's a resume/interview
project for an AI/ML engineering role. Every decision needs a defensible answer to
"why did you do it this way, and why not the obvious alternative." That framing
shapes everything below.

Approved plan (from initial scoping, 2026-07-13): smart re-crop (not generative
outpainting, not critique-only), train/fine-tune own models eventually (Model A: a
GAICD-trained crop scorer; Model B: a NIMA-style aesthetic model trained on AVA), web
app (FastAPI + React), ~1–2 month timeline across 4 phases, deploy to a free host with
a shareable URL. Full original plan: `docs/PLAN.md` in this repo.

---

## 2. How we work together — read this before doing anything

These are direct, explicit instructions from the user. Follow them without being asked
again:

1. **Teach, don't just implement.** Explain every decision: what we did, why, and why
   *not* the obvious alternative. The user is learning ML/CV/MLOps as we go and needs
   to be able to defend every choice to an interviewer. When something is genuinely
   uncertain (e.g. "will this Docker build fit in 512MB?"), say so honestly instead of
   projecting false confidence — a wrong confident guess wastes more of the user's
   time than an honest "I'm not sure, let's verify."
2. **ELI5 on request, with concrete examples.** If asked to re-explain something
   simply, actually simplify — analogies, small numbers, no unexplained jargon — don't
   just repeat the same explanation with the same vocabulary.
3. **Ask interview-style "grill" questions** after teaching something, and expect the
   user to attempt answers before being given them. If the user says they don't know
   the domain (this happened with computer vision), build up from zero rather than
   assuming prior knowledge.
4. **TDD, always.** Write tests first, show/confirm the red failure, then implement
   to green, then refactor/lint. Never write implementation before its test exists.
5. **Clean code + SOLID.** Point out where SOLID principles actually show up in the
   design (not just as buzzwords) — e.g. constraints as strategy objects behind a
   `Protocol` (open/closed, dependency inversion), single-responsibility modules.
6. **Tiny, single-purpose commits.** Every discrete step gets its own local commit
   with a message explaining the *why*, not just the what. The git log is meant to
   read as the project's build diary. Commit locally as you go; the user pushes to
   remotes deliberately (see below), not automatically on every commit.
7. **Never add `Co-Authored-By` or any AI-attribution trailer to commits.** This must
   read as the user's own independent project. Plain messages, user's own git
   identity only. (This was an explicit, firm correction early on — don't reintroduce
   it.)
8. **Don't take shortcuts that are technically worse to save time.** If there's a
   right way and a fast-but-wrong way, do the right way and explain the tradeoff
   rather than silently picking convenience.
9. **Regularly ask questions instead of assuming** — use multiple-choice-style
   decision points for real forks (e.g. deployment target, fix approach for a bug)
   rather than guessing at preference.
10. **Verify before shipping when possible.** Several bugs in this project were only
    caught because we ran the real pipeline / built the real Docker context / tested
    a sync command in isolation before pushing to a service with a slower feedback
    loop (a cloud build queue). Prefer local verification over "should work."
11. **Frame this as a real 2026 product**, not a toy: modern stack choices, a clear
    answer to "why not just call a vision LLM," MLOps polish (Docker, CI, ONNX,
    latency numbers) as a first-class part of the resume story, not an afterthought.

---

## 3. Current status (verified, as of this handoff)

- **Phase 1 (rule-based baseline pipeline) is complete.** Perception → grid-anchor
  candidate generation → hard constraints → five-rule composition scorer → FastAPI →
  React frontend. Verified end-to-end against a real photo, not just unit tests.
- **134 tests, all passing.** Zero tests require downloading a model (perception
  models sit behind `Protocol`s; tests inject fakes; YOLO pre/post-processing is
  tested against synthetic tensors). Run: `uv run pytest`.
- **Lint clean.** `uv run ruff check .` — zero suppressions; every finding was fixed,
  not silenced.
- **GitHub is up to date.** `main` locally matches `origin/main` exactly at `92414d6`.
  No other local branches exist (an `hf-space` branch was created, then deliberately
  deleted — see §6).
- **Deployment blocker is fixed locally (2026-07-16), pending push + Render
  verification.** See §5 — the OOM root cause is resolved and measured under
  budget; the remaining step is pushing `main` and watching the Render deploy.
- **Phase 2 (learned crop scorer on GAICD) has not started.** No dataset downloaded,
  no training code written yet. This is the natural next step once deployment is
  resolved or explicitly deprioritized.

Full architecture, API contract, and per-rule scoring formulas are documented in depth
in [README.md](README.md) — that file is the technical reference; this handoff is the
process/status reference. Don't duplicate effort re-deriving what's already written
there.

---

## 4. What's been built — condensed map

```
recompose/            core library (no web concerns), fully unit-tested
├── perception/       saliency (U²-Net via onnxruntime), detection (YOLOv8n via
│                      ultralytics — see §5, this is what's blocking deploy),
│                      horizon (OpenCV Hough lines)
├── cropping/         grid-anchor candidate generator (GAIC paper discretization:
│                      5 scales × 5×5 positions × 4 aspect ratios ≈ 500 candidates),
│                      constraint strategy objects (subject integrity, saliency
│                      retention) behind a `CropConstraint` Protocol
├── scoring/           five composition rules (thirds, retention, balance, headroom,
│                      subject_size), weighted-average fusion, `None` for
│                      inapplicable rules (excluded from the weight denominator,
│                      not scored as 0)
├── saliency_stats.py  summed-area tables (integral images) for O(1) region-mass and
│                      region-centroid queries — the Viola-Jones trick
├── pipeline.py         orchestrates the above; models are injected (dependency
│                      inversion); has a constraint-relaxation ladder so it never
│                      returns nothing, even for pathological aspect ratios
└── types.py            Box, CropCandidate, ScoredCrop — dependency-free geometry

api/                   FastAPI: schemas (Pydantic response models, all geometry
                       normalized to [0,1] fractions), imaging helpers (EXIF-aware
                       decode, downscale, saliency PNG encoding), app factory
                       (`create_app`, testable via injected fake pipeline + static
                       dir), production entrypoint (`api/server.py`)

web/                   React + Vite frontend: upload, aspect tabs, before/after
                       views, toggleable overlays (subject boxes, saliency heatmap,
                       crop outline, thirds grid), per-rule score bars, PNG download
                       at native resolution

tests/                 119 tests, TDD throughout, zero model downloads required

Dockerfile             two-stage build (Node builds frontend, Python serves both
                       API and static frontend from one process/port) — see §5 for
                       its current runtime memory problem
docs/PLAN.md           the original approved project plan
```

### The most important design decisions, condensed (full detail + rationale in README)

- **Boxes (YOLO) are hard constraints; saliency (U²-Net) is a soft signal.** A crop
  may never slice a detected subject; among crops that don't, saliency-based scoring
  picks the best one. Guardrail + ranker — the standard structure for production ML
  systems.
- **Saliency stays continuous, never thresholded to binary.** A weighted centroid and
  retention score both need to distinguish "definitely the subject" from "maybe
  background" — quantizing early throws away information every downstream consumer
  then lacks.
- **Grid-anchor candidates, not exhaustive search.** Composition quality varies
  smoothly with crop coordinates, so ~500 coarsely-sampled candidates cover the
  meaningfully-different framings without the ~10¹³ possible rectangles. This is also
  architecturally what the future learned model (Phase 2) will rank — the candidate
  generator doesn't change, only the judge does.
- **`None` ≠ 0 for inapplicable scoring rules.** Headroom/subject-size rules return
  `None` (not 0) when there's no subject, and `None` results are excluded from the
  weighted-average denominator — so a landscape photo isn't unfairly penalized by
  rules that have no opinion about it.
- **Constraint relaxation ladder, never returns nothing.** If strict filtering leaves
  zero candidates (e.g. a 16:9 crop of a portrait-orientation photo structurally can't
  keep 55% of the image), constraints relax in value order: retention before subject
  integrity, because a loose crop is a taste problem while an amputated subject is a
  defect. The API surfaces a `constraints_relaxed` flag so the UI can be honest.
- **All API geometry is normalized to [0,1] fractions**, not pixels — the server
  analyzes a downscaled copy, the browser displays at any size; fractions make every
  consumer resolution-independent and eliminate an entire class of "whose pixels?"
  overlay bugs.
- **Models load once at FastAPI startup (lifespan), not per-request or lazily.**
  Failures surface at deploy time, not on a user's request; readiness probes reflect
  true serving ability.

---

## 5. Deployment — current blocker (pick up here first)

**Target: Render** (chosen deliberately — free, no credit card required, deploys
directly from our `Dockerfile` on GitHub with zero code changes needed for that part).

We first tried Hugging Face Spaces, which was the original plan's Phase-4 target, but
discovered mid-session that **HF now requires a PRO subscription to create Docker (or
even Gradio) Spaces on the free `cpu-basic` tier** — an undocumented, recent policy
change we confirmed via HF's own community forum, not something the user did wrong.
Static Spaces are still free but can't run our Python backend at all. We pivoted to
Render without hesitation once this was confirmed. (There's a leftover mental note:
if the user later gets HF PRO or the policy reverts, recreating an `hf-space` branch
is trivial — just add YAML front matter to `README.md`'s top; we did this once, then
deleted the branch since it was unused. No code loss either way.)

### Bugs hit and fixed during the Render deploy attempt (in order — don't re-diagnose these)

1. **Render initially built from a stale commit** ("no such file: Dockerfile," and
   auto-detected Python instead of Docker). Root cause: `git push origin main` had
   only been run once, *before* the Dockerfile/static-serving work was committed
   locally — GitHub's `main` was several commits behind local `main`. Fixed by
   pushing again. **Lesson: after connecting a repo to an auto-deploy service,
   double check the remote actually has the commit you think it does before blaming
   the service.**
2. **`uv sync` failed inside the Docker build**: `OSError: Readme file does not
   exist: README.md`. Root cause: `pyproject.toml` declares `readme = "README.md"`,
   but the Dockerfile only copied `pyproject.toml` + `uv.lock` before running
   `uv sync` (deliberately, for Docker layer caching) — so hatchling's build-backend
   metadata validation failed looking for a file that wasn't copied yet. Fixed with
   the officially-recommended uv Docker pattern: `uv sync --frozen --no-install-project`
   first (installs only third-party deps, no local package build, so no README
   needed), then copy source, then a second `uv sync --frozen` to install the local
   project (fast, since deps are already cached). Verified in isolation (a scratch
   directory with only the three dependency files) before pushing, since Docker
   itself isn't installed in this dev environment and the image can't be built
   locally to double check.
3. **A branch-management mistake**: while debugging, an `hf-space` branch had been
   created earlier and never switched back from — so the Dockerfile fix commit
   initially landed on the wrong branch and `git push origin main` silently no-opped
   ("Everything up-to-date") because `main` hadn't actually changed. Caught by
   comparing `git log origin/main` vs `git log main` directly rather than trusting
   the push output. Fixed via `git checkout main && git cherry-pick <fix-commit>`,
   then rebuilt `hf-space` cleanly on top (later deleted entirely once Render was
   confirmed working and HF was deprioritized).
4. **RESOLVED (2026-07-16): runtime OOM.** The container died with `Out of memory
   (used over 512Mi)` because `ultralytics` transitively installed CUDA-enabled
   PyTorch. Fixed with the ONNX rewrite (the recommended option), plus two more
   levers that local peak-RSS measurement proved necessary:
   - `SubjectDetector` now runs `yolov8n.onnx` on pure `onnxruntime` with
     hand-rolled letterbox/decode/NMS (tested against synthetic tensors; verified
     against the old ultralytics path at IoU 0.96–0.99 on real photos).
     `ultralytics`+`onnx` moved to an `export` extra; the Dockerfile grew a
     throwaway weights stage. Torch never enters the runtime image.
   - That alone was NOT enough: a 20ms RSS sampler against the real server showed
     555MB peak during the first request. onnxruntime's CPU memory arena retains
     ~240MB of activation buffers forever (disabled), each extra thread holds
     scratch buffers (sessions pinned to 1 thread — Render free tier has 0.1 CPU
     anyway), and **full U²-Net's first inference peaks over 512MB entirely on its
     own** — so the default saliency model is now u2netp (4.4MB), with
     `RECOMPOSE_SALIENCY_MODEL=u2net` restoring the full model on bigger hardware.
     Crop quality: 3 of 4 aspect ratios chose identical crops under both models on
     the reference photo.
   - **Measured final footprint: 421MB peak / 200MB steady / 92MB startup**,
     ~0.85s per analyze request, single-threaded, on macOS (Linux will differ
     somewhat — watch the first Render deploy).
   - CMD honors Render's injected `PORT` env var now (shell-form CMD).

---

## 6. Other lessons already learned (don't rediscover these)

- **`uv` dependency resolution picked Python 3.14 by default**, which broke building
  `rembg`'s transitive chain (`pymatting → numba → llvmlite`, a 2021-era llvmlite that
  doesn't declare its Python ceiling correctly). Fixed by pinning `.python-version` to
  3.12. More importantly, we then **removed `rembg` entirely** — it was only ever a
  40-line convenience wrapper around downloading `u2net.onnx` and running it through
  `onnxruntime`, both of which we already needed directly. Lesson demonstrated: before
  fighting a dependency's transitive chain, ask whether you need the dependency at
  all.
- **OpenCV's `HoughLinesP` output shape drifted** between versions (`(N,1,4)` vs
  `(N,4)`), crashing horizon estimation with `TypeError: cannot unpack non-iterable
  numpy.int32 object` — caught only by the end-to-end run against a real photo, not
  by the 108 unit tests that existed at the time (horizon estimation was the one
  perception module without direct tests). Fixed with `lines.reshape(-1, 4)` plus new
  regression tests that run the real OpenCV path against synthetic tilted-line scenes
  with known ground-truth angles. Lesson: unit tests verify your logic; integration
  tests verify your assumptions about everyone else's library behavior.
- **`CandidateGridConfig()` as a function-call default argument** was flagged by ruff
  (B008) — Python evaluates default arguments once at import time, a classic
  mutable-default footgun. Fixed with the `param: X | None = None` then
  `param = param or X()` idiom.
- **FastAPI's `File(...)` / `Query(...)` as bare defaults** hit the same B008 class of
  lint issue; fixed with `Annotated[Type, File()]` / `Annotated[Type, Query(...)]`,
  the modern FastAPI-recommended style.

---

## 7. What's next

Immediate: resolve the deployment blocker in §5 (ask the user which fix they want),
or explicitly deprioritize deployment and move to Phase 2 if the user prefers to make
ML progress first — both are reasonable; don't assume, ask.

**Phase 2 (not started): the learned crop scorer.**
- Download GAICD (Grid Anchor based Image Cropping Database: ~1,236 images × ~86
  human-rated candidate crops each).
- Write a `torch.utils.data.Dataset` loader.
- Model: shared CNN backbone + RoIAlign over candidate regions + a small scoring
  head (the actual GAIC architecture) — the same trick that turned R-CNN into Fast
  R-CNN: run the expensive backbone once, score all candidates against one shared
  feature map.
- Training loop: PyTorch + timm, mixed precision, cosine LR schedule, Weights &
  Biases experiment tracking, config-driven (Hydra or plain YAML).
- Evaluation: SRCC (rank correlation) and Acc@K against the GAICD test split,
  reported against published baselines (GAIC, CGS, TransView).
- **The headline ablation**: this learned scorer vs. the Phase 1 rule-based baseline
  we already built and shipped — hand-tuned heuristics vs. learned scoring, with real
  numbers on both sides.

Phase 3 (NIMA-style aesthetic model on AVA) and Phase 4 (ship: ONNX export,
quantization, CI, public demo) follow after that — full detail in `docs/PLAN.md`.

---

## 8. Quick reference

```bash
# backend (Python 3.12 via uv; first run downloads U²-Net-p ~4MB and exports
# yolov8n.onnx via the export extra - hence --all-extras for dev)
uv sync --all-extras
uv run pytest                                 # 134 tests, no downloads needed
uv run ruff check .                           # zero suppressions, must stay clean
uv run uvicorn api.server:app --port 8000     # serves API + built frontend if
                                               # web/dist exists (npm run build)

# frontend dev server (hot reload, proxies /api to :8000)
cd web && npm install && npm run dev          # http://localhost:5173
```

- GitHub: [github.com/lokeshgupta22/ReCompose](https://github.com/lokeshgupta22/ReCompose)
- Render: connected to this repo's `main` branch, auto-deploys on push (currently
  failing to *run* post-build per §5; the build itself succeeds)
- Hugging Face username on file: `lokeshgupta22` (Docker Spaces currently paywalled
  behind PRO for this account — see §5)
- Model weight cache at runtime: `~/.cache/recompose` (override via
  `RECOMPOSE_CACHE_DIR` env var)
