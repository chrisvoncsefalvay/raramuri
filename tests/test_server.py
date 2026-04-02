"""Hot server health and response tests.

Tests that infer_server.py starts correctly, responds to health checks,
and produces valid inference results. Designed to run against a server
that is already running in a container.

Set RARAMURI_SERVER_URL to point to the running server, e.g.:

    RARAMURI_SERVER_URL=http://localhost:8876 python -m pytest tests/test_server.py -v
"""

import json
import os

import pytest
import urllib.request
import urllib.error

SERVER_URL = os.environ.get("RARAMURI_SERVER_URL", "http://localhost:8876")
FIXTURE_VIDEO = os.environ.get(
    "RARAMURI_TEST_VIDEO", "/repo/profiling/bluey-15s-90s.mp4"
)


def _get(path: str) -> tuple[int, dict]:
    """GET request, return (status, parsed JSON)."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except urllib.error.URLError:
        pytest.skip(f"Server not reachable at {SERVER_URL}")


def _get_text(path: str) -> tuple[int, str]:
    """GET request, return (status, raw text)."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except urllib.error.URLError:
        pytest.skip(f"Server not reachable at {SERVER_URL}")


def _post(path: str, payload: dict) -> tuple[int, dict]:
    """POST request, return (status, parsed JSON)."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{SERVER_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except urllib.error.URLError:
        pytest.skip(f"Server not reachable at {SERVER_URL}")


class TestHealth:
    def test_health_endpoint(self):
        status, body = _get("/health")
        assert status == 200
        assert body.get("ok") is True
        assert "ready" in body

    def test_ready_endpoint(self):
        status, body = _get("/ready")
        assert status == 200
        assert body.get("ready") is True

    def test_status_endpoint(self):
        status, body = _get("/status")
        assert status == 200
        assert body.get("ready") is True
        assert "runtime" in body
        assert "current_request" in body
        assert "last_request" in body

    def test_metrics_endpoint(self):
        status, body = _get_text("/metrics")
        assert status == 200
        assert "raramuri_server_ready" in body
        assert "raramuri_server_active_requests" in body

    def test_unknown_path_returns_404(self):
        status, _ = _get("/nonexistent")
        assert status == 404


class TestInference:
    def test_infer_returns_predictions(self):
        status, body = _post("/infer", {"video_path": FIXTURE_VIDEO})
        assert status == 200
        assert "predictions" in body
        assert "shape" in body

    def test_prediction_shape(self):
        status, body = _post("/infer", {"video_path": FIXTURE_VIDEO})
        assert status == 200
        shape = body["shape"]
        assert len(shape) == 2
        assert shape[1] == 20484

    def test_timing_present(self):
        status, body = _post("/infer", {"video_path": FIXTURE_VIDEO})
        assert status == 200
        assert "timing" in body
        assert body["timing"]["total_seconds"] > 0

    def test_missing_video_returns_400(self):
        status, body = _post("/infer", {"video_path": "/nonexistent/video.mp4"})
        assert status == 400
        assert "error" in body
        assert body["code"] == "video_not_found"
