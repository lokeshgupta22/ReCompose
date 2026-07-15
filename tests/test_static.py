"""Contract for serving the built frontend alongside the API.

In production (the Docker image), FastAPI serves both /api/* and the
built React app from one process/port. static_dir is injectable so
these tests don't depend on `npm run build` having been run, and so
api.main stays importable with zero side effects when no build exists.
"""

from fastapi.testclient import TestClient

from api.main import create_app


class FakePipeline:
    def analyze(self, image, aspects, top_k):
        raise AssertionError("static-file tests should never reach the pipeline")


def build_client(static_dir=None):
    return TestClient(create_app(lambda: FakePipeline(), static_dir=static_dir))


class TestNoStaticDir:
    def test_root_is_404_when_no_frontend_is_configured(self):
        with build_client(static_dir=None) as client:
            assert client.get("/").status_code == 404

    def test_api_routes_still_work(self):
        with build_client(static_dir=None) as client:
            assert client.get("/api/health").status_code == 200


class TestWithStaticDir:
    def make_build(self, tmp_path):
        (tmp_path / "index.html").write_text("<html><body>recompose</body></html>")
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("console.log('hi')")
        return tmp_path

    def test_root_serves_index_html(self, tmp_path):
        static_dir = self.make_build(tmp_path)
        with build_client(static_dir=static_dir) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "recompose" in response.text

    def test_asset_files_are_served(self, tmp_path):
        static_dir = self.make_build(tmp_path)
        with build_client(static_dir=static_dir) as client:
            response = client.get("/assets/app.js")
            assert response.status_code == 200
            assert "console.log" in response.text

    def test_unknown_path_falls_back_to_index_html(self, tmp_path):
        # Single-page app: any non-API, non-asset path should still load
        # the app shell rather than 404.
        static_dir = self.make_build(tmp_path)
        with build_client(static_dir=static_dir) as client:
            response = client.get("/some/deep/link")
            assert response.status_code == 200
            assert "recompose" in response.text

    def test_api_routes_take_priority_over_static_catchall(self, tmp_path):
        static_dir = self.make_build(tmp_path)
        with build_client(static_dir=static_dir) as client:
            assert client.get("/api/health").json() == {"status": "ok"}
