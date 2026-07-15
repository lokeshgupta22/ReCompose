"""Uvicorn entry point: `uvicorn api.server:app`.

Kept separate from the factory so importing api.main (e.g. in tests) never
constructs the real model-backed app.
"""

from api.main import create_app

app = create_app()
