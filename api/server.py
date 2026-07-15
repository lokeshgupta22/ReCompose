"""Uvicorn entry point: `uvicorn api.server:app`.

Kept separate from the factory so importing api.main (e.g. in tests) never
constructs the real model-backed app. This is also the one place that opts
into serving the built frontend: create_app defaults static_dir to None so
test behavior never depends on whether `npm run build` has been run.
"""

from api.main import DEFAULT_STATIC_DIR, create_app

app = create_app(static_dir=DEFAULT_STATIC_DIR)
