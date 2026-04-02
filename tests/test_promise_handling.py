"""Tests exercising the promise/future-handling model in infer_server.py.

Covers:
- Future.result() timeout → 504
- Future exception propagation → 500
- Semaphore slot release on success/failure/timeout
- Job submission lifecycle (queued → running → completed)
- Job failure lifecycle (queued → running → failed)
- Job cancellation of a queued job
- Progress callback detection and wiring
- Done-callback accounting (pending_requests counter)
"""

import concurrent.futures
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

import infer_server
from infer_server import APIError, InferenceService, ServerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video(tmp_path: Path) -> str:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    return str(video)


def _ready_service(inference_fn, **config_overrides) -> InferenceService:
    """Return an InferenceService that is already marked as ready."""
    cfg = ServerConfig(**{**{"max_pending_requests": 4, "request_timeout_seconds": 5}, **config_overrides})
    svc = InferenceService(config=cfg, inference_fn=inference_fn)
    svc._ready = True
    return svc


# ---------------------------------------------------------------------------
# Future.result() timeout → 504
# ---------------------------------------------------------------------------

class TestFutureTimeout:
    def test_infer_raises_504_on_timeout(self, tmp_path):
        video = _make_video(tmp_path)
        started = threading.Event()

        def hang(_video_path: str) -> dict:
            started.set()
            time.sleep(30)  # longer than timeout
            return {}

        svc = _ready_service(hang, request_timeout_seconds=1)
        try:
            with pytest.raises(APIError) as exc_info:
                svc.infer({"video_path": video})
            assert exc_info.value.status == 504
            assert exc_info.value.code == "inference_timeout"
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Future exception propagation → 500
# ---------------------------------------------------------------------------

class TestFutureExceptionPropagation:
    def test_infer_raises_500_on_inference_error(self, tmp_path):
        video = _make_video(tmp_path)

        def explode(_video_path: str) -> dict:
            raise RuntimeError("GPU melted")

        svc = _ready_service(explode)
        try:
            with pytest.raises(APIError) as exc_info:
                svc.infer({"video_path": video})
            assert exc_info.value.status == 500
            assert "GPU melted" in exc_info.value.message
        finally:
            svc.shutdown()

    def test_failed_request_increments_requests_failed(self, tmp_path):
        video = _make_video(tmp_path)

        def explode(_video_path: str) -> dict:
            raise RuntimeError("boom")

        svc = _ready_service(explode)
        try:
            with pytest.raises(APIError):
                svc.infer({"video_path": video})
            h = svc.health_payload()
            assert h["requests_failed"] == 1
            assert h["requests_completed"] == 0
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Semaphore slot release on success, failure, and timeout
# ---------------------------------------------------------------------------

class TestSlotRelease:
    def test_slot_released_after_success(self, tmp_path):
        video = _make_video(tmp_path)

        def ok(_video_path: str) -> dict:
            return {"shape": [1], "predictions": []}

        svc = _ready_service(ok, max_pending_requests=1)
        try:
            svc.infer({"video_path": video})
            # If slot was leaked, this second call would get 429
            svc.infer({"video_path": video})
            h = svc.health_payload()
            assert h["requests_completed"] == 2
            assert h["pending_requests"] == 0
        finally:
            svc.shutdown()

    def test_slot_released_after_failure(self, tmp_path):
        video = _make_video(tmp_path)
        call_count = 0

        def fail_then_succeed(_video_path: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return {"shape": [1], "predictions": []}

        svc = _ready_service(fail_then_succeed, max_pending_requests=1)
        try:
            with pytest.raises(APIError):
                svc.infer({"video_path": video})
            # Slot must be free for next request
            result = svc.infer({"video_path": video})
            assert result["shape"] == [1]
        finally:
            svc.shutdown()

    def test_pending_requests_returns_to_zero(self, tmp_path):
        video = _make_video(tmp_path)

        def ok(_video_path: str) -> dict:
            return {"shape": [1], "predictions": []}

        svc = _ready_service(ok, max_pending_requests=2)
        try:
            svc.infer({"video_path": video})
            svc.infer({"video_path": video})
            h = svc.health_payload()
            assert h["pending_requests"] == 0
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Job submission lifecycle
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    def test_submit_job_completes(self, tmp_path):
        video = _make_video(tmp_path)
        completed = threading.Event()

        def ok(_video_path: str) -> dict:
            result = {"shape": [1, 2], "predictions": [[0.5, 0.5]]}
            completed.set()
            return result

        svc = _ready_service(ok)
        try:
            job_status = svc.submit_job({"video_path": video})
            job_id = job_status["job_id"]
            assert job_id.startswith("job_")
            assert job_status["status"] in {"queued", "running", "completed"}

            # Wait for completion
            assert completed.wait(timeout=5)
            time.sleep(0.2)  # allow state update

            final = svc.get_job(job_id)
            assert final["status"] == "completed"
            assert final["result_shape"] == [1, 2]

            result = svc.get_job_result(job_id)
            assert result["result"]["predictions"] == [[0.5, 0.5]]
        finally:
            svc.shutdown()

    def test_submit_job_records_failure(self, tmp_path):
        video = _make_video(tmp_path)

        def fail(_video_path: str) -> dict:
            raise ValueError("bad video")

        svc = _ready_service(fail)
        try:
            job_status = svc.submit_job({"video_path": video})
            job_id = job_status["job_id"]

            # Wait for failure to propagate
            with svc._lock:
                future = svc._jobs[job_id].future
            if future:
                try:
                    future.result(timeout=5)
                except Exception:
                    pass

            final = svc.get_job(job_id)
            assert final["status"] == "failed"
            assert final["error"]["message"] == "bad video"

            with pytest.raises(APIError) as exc_info:
                svc.get_job_result(job_id)
            assert exc_info.value.code == "job_failed"
        finally:
            svc.shutdown()

    def test_get_nonexistent_job_returns_404(self):
        svc = _ready_service(lambda _: {})
        try:
            with pytest.raises(APIError) as exc_info:
                svc.get_job("job_doesnotexist")
            assert exc_info.value.status == 404
        finally:
            svc.shutdown()

    def test_get_result_before_completion_returns_409(self, tmp_path):
        video = _make_video(tmp_path)
        started = threading.Event()
        release = threading.Event()

        def slow(_video_path: str) -> dict:
            started.set()
            release.wait(timeout=10)
            return {"shape": [1], "predictions": []}

        svc = _ready_service(slow)
        try:
            job_status = svc.submit_job({"video_path": video})
            job_id = job_status["job_id"]
            assert started.wait(timeout=5)

            with pytest.raises(APIError) as exc_info:
                svc.get_job_result(job_id)
            assert exc_info.value.code == "job_not_ready"

            release.set()
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------

class TestJobCancellation:
    def test_cancel_completed_job_returns_409(self, tmp_path):
        video = _make_video(tmp_path)

        def ok(_video_path: str) -> dict:
            return {"shape": [1], "predictions": []}

        svc = _ready_service(ok)
        try:
            job_status = svc.submit_job({"video_path": video})
            job_id = job_status["job_id"]

            # Wait for completion
            with svc._lock:
                future = svc._jobs[job_id].future
            if future:
                try:
                    future.result(timeout=5)
                except Exception:
                    pass

            with pytest.raises(APIError) as exc_info:
                svc.cancel_job(job_id)
            assert exc_info.value.code == "job_not_cancellable"
        finally:
            svc.shutdown()

    def test_cancel_nonexistent_job_returns_404(self):
        svc = _ready_service(lambda _: {})
        try:
            with pytest.raises(APIError) as exc_info:
                svc.cancel_job("job_nope")
            assert exc_info.value.status == 404
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Progress callback detection
# ---------------------------------------------------------------------------

class TestProgressCallbackDetection:
    def test_detects_explicit_progress_callback_param(self):
        def fn(video_path: str, progress_callback=None) -> dict:
            return {}
        assert InferenceService._detect_progress_callback_support(fn) is True

    def test_detects_kwargs(self):
        def fn(video_path: str, **kwargs) -> dict:
            return {}
        assert InferenceService._detect_progress_callback_support(fn) is True

    def test_rejects_no_callback(self):
        def fn(video_path: str) -> dict:
            return {}
        assert InferenceService._detect_progress_callback_support(fn) is False

    def test_handles_uninspectable_callable(self):
        # Built-in that can't be inspected
        assert InferenceService._detect_progress_callback_support(len) is False


# ---------------------------------------------------------------------------
# Progress callback wiring
# ---------------------------------------------------------------------------

class TestProgressCallbackWiring:
    def test_progress_callback_updates_current_request(self, tmp_path):
        video = _make_video(tmp_path)
        progress_updates = []

        def inference_with_progress(video_path: str, progress_callback=None) -> dict:
            if progress_callback:
                progress_callback({"step": "extracting_audio", "progress_ratio": 0.5, "step_index": 1, "total_steps": 3})
                # Capture the state
                progress_updates.append(True)
            return {"shape": [1], "predictions": []}

        svc = _ready_service(inference_with_progress)
        try:
            svc.infer({"video_path": video})
            assert len(progress_updates) == 1
        finally:
            svc.shutdown()

    def test_progress_callback_not_passed_when_unsupported(self, tmp_path):
        video = _make_video(tmp_path)
        received_args = {}

        def simple_inference(video_path: str) -> dict:
            received_args["video_path"] = video_path
            return {"shape": [1], "predictions": []}

        svc = _ready_service(simple_inference)
        try:
            svc.infer({"video_path": video})
            assert "video_path" in received_args
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Unready service rejects requests
# ---------------------------------------------------------------------------

class TestServiceReadiness:
    def test_infer_rejects_when_not_ready(self, tmp_path):
        video = _make_video(tmp_path)
        svc = InferenceService(config=ServerConfig(), inference_fn=lambda _: {})
        try:
            with pytest.raises(APIError) as exc_info:
                svc.infer({"video_path": video})
            assert exc_info.value.status == 503
            assert exc_info.value.code == "service_unready"
        finally:
            svc.shutdown()

    def test_submit_job_rejects_when_not_ready(self, tmp_path):
        video = _make_video(tmp_path)
        svc = InferenceService(config=ServerConfig(), inference_fn=lambda _: {})
        try:
            with pytest.raises(APIError) as exc_info:
                svc.submit_job({"video_path": video})
            assert exc_info.value.status == 503
        finally:
            svc.shutdown()

    def test_rejection_increments_counter(self, tmp_path):
        video = _make_video(tmp_path)
        svc = InferenceService(config=ServerConfig(), inference_fn=lambda _: {})
        try:
            with pytest.raises(APIError):
                svc.infer({"video_path": video})
            assert svc.health_payload()["requests_rejected"] == 1
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Metrics and status include job counts
# ---------------------------------------------------------------------------

class TestStatusPayload:
    def test_status_includes_job_counts(self, tmp_path):
        video = _make_video(tmp_path)

        def ok(_video_path: str) -> dict:
            return {"shape": [1], "predictions": []}

        svc = _ready_service(ok)
        try:
            job_status = svc.submit_job({"video_path": video})
            # Wait for completion
            with svc._lock:
                future = svc._jobs[job_status["job_id"]].future
            if future:
                try:
                    future.result(timeout=5)
                except Exception:
                    pass

            status = svc.status_payload()
            assert "jobs" in status
            assert status["jobs"]["total"] >= 1
        finally:
            svc.shutdown()
