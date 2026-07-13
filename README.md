# ReCompose

**Upload a photo. Get it professionally composed.**

ReCompose finds the subject of your photo, understands its composition (rule of thirds,
headroom, balance, horizon), and returns the best crop for every aspect ratio — along
with a visual explanation of *why* that crop works. Every photographer gets a
composition assistant; nobody has to memorize the rules.

Unlike sending your photos to a cloud VLM, ReCompose runs a purpose-built pipeline:
deterministic, explainable, fast enough for CPU, and your photos never need to leave
the machine.

## How it works

```
photo → [1. Perception] → [2. Candidate crops] → [3. Scoring & ranking] → best crop + why
```

1. **Perception** — U²-Net saliency (where the visual mass is), YOLOv8 subject detection
   (hard constraints: never amputate a subject), Hough-based horizon tilt estimation.
2. **Candidate generation** — grid-anchor crop candidates across aspect ratios, filtered
   by subject-safety and saliency-retention constraints.
3. **Scoring** — composition scoring (rule-based baseline; learned crop scorer trained on
   GAICD and a NIMA-style aesthetic model trained on AVA arrive in later phases).

## Status

Phase 1 (rule-based baseline + web app) — in progress.

## Development

```bash
uv sync --all-extras   # install
uv run pytest          # test
```
