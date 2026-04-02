import json
import threading
from pathlib import Path
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

import infer_server


def _get_json(base_url: str, path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base_url}{path}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get_text(base_url: str, path: str) -> tuple[int, str]:
    req = urllib.request.Request(f"{base_url}{path}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8")


def _post_json(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def runtime_snapshot(monkeypatch):
    snapshot = {
        "process": {
            "rss_max_mb": 256.0,
            "rss_current_mb": 192.0,
            "vmsize_mb": 1024.0,
        },
        "gpu": {
            "allocated_mb": 512.0,
            "reserved_mb": 768.0,
            "max_allocated_mb": 1024.0,
            "max_reserved_mb": 1536.0,
        },
    }
    monkeypatch.setattr(infer_server, "runtime_status_snapshot", lambda: snapshot)
    return snapshot


def _start_server(inference_fn):
    service = infer_server.InferenceService(
        config=infer_server.ServerConfig(max_pending_requests=1, request_timeout_seconds=5),
        inference_fn=inference_fn,
    )
    service._ready = True
    server = infer_server.InferenceHTTPServer(("127.0.0.1", 0), infer_server.Handler, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return base_url, service, server, thread


def test_status_endpoint_reports_live_request_progress(tmp_path, runtime_snapshot):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    entered = threading.Event()
    release = threading.Event()
    result = {}

    def slow_inference(_video_path: str, progress_callback=None) -> dict:
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "event_build",
                    "stage": "started",
                    "step_index": 2,
                    "total_steps": 6,
                    "completed_steps": 1,
                    "progress_ratio": 1 / 6,
                    "elapsed_seconds": 0.25,
                    "step_elapsed_seconds": 0.1,
                    "runtime": runtime_snapshot,
                }
            )
        entered.set()
        release.wait(timeout=5)
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "complete",
                    "stage": "completed",
                    "step_index": 6,
                    "total_steps": 6,
                    "completed_steps": 6,
                    "progress_ratio": 1.0,
                    "elapsed_seconds": 0.5,
                    "step_elapsed_seconds": 0.5,
                    "runtime": runtime_snapshot,
                }
            )
        return {"shape": [1, 2], "predictions": [[0.0, 1.0]], "timing": {"total_seconds": 0.5}}

    base_url, service, server, thread = _start_server(slow_inference)
    try:
        request_thread = threading.Thread(
            target=lambda: result.update(value=_post_json(base_url, "/infer", {"video_path": str(video)})),
            daemon=True,
        )
        request_thread.start()

        assert entered.wait(timeout=2), "inference request did not reach fake backend"

        status_code, status_body = _get_json(base_url, "/status")
        assert status_code == 200
        assert status_body["ready"] is True
        assert status_body["current_request"]["step"] == "event_build"
        assert status_body["current_request"]["progress_ratio"] == pytest.approx(1 / 6)
        assert status_body["current_request"]["runtime"]["gpu"]["allocated_mb"] == 512.0
        assert status_body["runtime"]["process"]["rss_current_mb"] == 192.0

        release.set()
        request_thread.join(timeout=5)
        assert result["value"][0] == 200

        status_code, status_body = _get_json(base_url, "/status")
        assert status_code == 200
        assert status_body["current_request"] is None
        assert status_body["last_request"]["status"] == "completed"
        assert status_body["last_request"]["result_shape"] == [1, 2]
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()
        thread.join(timeout=5)


def test_metrics_endpoint_exposes_runtime_and_request_counters(tmp_path, runtime_snapshot):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    entered = threading.Event()
    release = threading.Event()

    def slow_inference(_video_path: str, progress_callback=None) -> dict:
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "predict",
                    "stage": "started",
                    "step_index": 4,
                    "total_steps": 6,
                    "completed_steps": 3,
                    "progress_ratio": 0.5,
                    "elapsed_seconds": 1.0,
                    "step_elapsed_seconds": 0.2,
                    "runtime": runtime_snapshot,
                }
            )
        entered.set()
        release.wait(timeout=5)
        return {"shape": [1, 1], "predictions": [[0.0]], "timing": {"total_seconds": 1.0}}

    base_url, service, server, thread = _start_server(slow_inference)
    try:
        request_thread = threading.Thread(
            target=lambda: _post_json(base_url, "/infer", {"video_path": str(video)}),
            daemon=True,
        )
        request_thread.start()
        assert entered.wait(timeout=2), "inference request did not reach fake backend"

        status_code, metrics_body = _get_text(base_url, "/metrics")
        assert status_code == 200
        assert "raramuri_server_ready 1" in metrics_body
        assert "raramuri_server_requests_started_total 1" in metrics_body
        assert "raramuri_server_current_request_step_index 4" in metrics_body
        assert "raramuri_server_current_request_progress_ratio 0.5" in metrics_body
        assert "raramuri_server_gpu_allocated_mb 512.0" in metrics_body
        assert "raramuri_server_process_rss_current_mb 192.0" in metrics_body

        release.set()
        request_thread.join(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()
        thread.join(timeout=5)


def test_job_api_submit_poll_and_result(tmp_path, runtime_snapshot):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    entered = threading.Event()
    release = threading.Event()

    def slow_inference(_video_path: str, progress_callback=None) -> dict:
        if progress_callback is not None:
            progress_callback(
                {
                    "step": "event_build",
                    "stage": "started",
                    "step_index": 2,
                    "total_steps": 6,
                    "completed_steps": 1,
                    "progress_ratio": 1 / 6,
                    "elapsed_seconds": 0.25,
                    "step_elapsed_seconds": 0.1,
                    "runtime": runtime_snapshot,
                }
            )
        entered.set()
        release.wait(timeout=5)
        return {"shape": [1, 2], "predictions": [[0.0, 1.0]], "timing": {"total_seconds": 0.5}}

    base_url, service, server, thread = _start_server(slow_inference)
    try:
        status_code, body = _post_json(base_url, "/jobs", {"video_path": str(video)})
        assert status_code == 202
        assert body["job_id"].startswith("job_")
        assert body["status"] in {"queued", "running"}
        assert body["status_url"].endswith(body["job_id"])
        assert body["result_url"].endswith(f"{body['job_id']}/result")

        job_id = body["job_id"]
        assert entered.wait(timeout=2), "job did not reach fake backend"

        status_code, status_body = _get_json(base_url, f"/jobs/{job_id}")
        assert status_code == 200
        assert status_body["job_id"] == job_id
        assert status_body["status"] == "running"
        assert status_body["phase"] == "event_build"
        assert status_body["progress_ratio"] == pytest.approx(1 / 6)

        status_code, result_body = _get_json(base_url, f"/jobs/{job_id}/result")
        assert status_code == 409
        assert result_body["code"] == "job_not_ready"

        release.set()

        for _ in range(20):
            status_code, status_body = _get_json(base_url, f"/jobs/{job_id}")
            if status_body["status"] == "completed":
                break
            time.sleep(0.05)
        assert status_body["status"] == "completed"
        assert status_body["result_shape"] == [1, 2]

        status_code, result_body = _get_json(base_url, f"/jobs/{job_id}/result")
        assert status_code == 200
        assert result_body["job_id"] == job_id
        assert result_body["result"]["shape"] == [1, 2]
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()
        thread.join(timeout=5)


def test_job_api_can_cancel_queued_job(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    release = threading.Event()

    def slow_inference(_video_path: str, progress_callback=None) -> dict:
        release.wait(timeout=5)
        return {"shape": [1, 1], "predictions": [[0.0]], "timing": {"total_seconds": 1.0}}

    service = infer_server.InferenceService(
        config=infer_server.ServerConfig(max_pending_requests=2, request_timeout_seconds=5),
        inference_fn=slow_inference,
    )
    service._ready = True
    server = infer_server.InferenceHTTPServer(("127.0.0.1", 0), infer_server.Handler, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        status_code, first_job = _post_json(base_url, "/jobs", {"video_path": str(video)})
        assert status_code == 202

        status_code, second_job = _post_json(base_url, "/jobs", {"video_path": str(video)})
        assert status_code == 202

        status_code, cancel_body = _post_json(base_url, f"/jobs/{second_job['job_id']}/cancel", {})
        assert status_code == 200
        assert cancel_body["status"] == "cancelled"

        status_code, status_body = _get_json(base_url, f"/jobs/{second_job['job_id']}")
        assert status_code == 200
        assert status_body["status"] == "cancelled"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        service.shutdown()
        thread.join(timeout=5)
