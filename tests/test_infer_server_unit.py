import threading
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

import infer_server


def test_validate_payload_accepts_existing_local_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    payload = infer_server.InferenceService._validate_payload({"video_path": str(video)})

    assert payload == {"video_path": str(video)}


def test_validate_payload_rejects_unknown_fields(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    with pytest.raises(infer_server.APIError) as exc_info:
        infer_server.InferenceService._validate_payload(
            {"video_path": str(video), "unexpected": True}
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "unknown_fields"


def test_validate_payload_accepts_remote_url_with_time_range():
    payload = infer_server.InferenceService._validate_payload(
        {
            "video_url": "https://www.youtube.com/watch?v=f7NwyBnIRTE",
            "start_time": "00:00:00",
            "end_time": "00:00:30",
        }
    )

    assert payload == {
        "video_url": "https://www.youtube.com/watch?v=f7NwyBnIRTE",
        "start_time": "00:00:00",
        "end_time": "00:00:30",
    }


def test_validate_payload_rejects_remote_path_in_video_path():
    with pytest.raises(infer_server.APIError) as exc_info:
        infer_server.InferenceService._validate_payload({"video_path": "https://example.com/video.mp4"})

    assert exc_info.value.status == 400
    assert exc_info.value.code == "remote_video_unsupported"


def test_validate_payload_rejects_invalid_time_range():
    with pytest.raises(infer_server.APIError) as exc_info:
        infer_server.InferenceService._validate_payload(
            {
                "video_url": "https://www.youtube.com/watch?v=f7NwyBnIRTE",
                "start_time": "30",
                "end_time": "10",
            }
        )

    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_time_range"


def test_infer_rejects_concurrent_request_when_capacity_is_full(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    started = threading.Event()
    release = threading.Event()

    def slow_inference(_video_path: str) -> dict:
        started.set()
        release.wait(timeout=5)
        return {"shape": [1, 1], "predictions": [[0.0]]}

    service = infer_server.InferenceService(
        config=infer_server.ServerConfig(max_pending_requests=1, request_timeout_seconds=5),
        inference_fn=slow_inference,
    )
    service._ready = True

    first_result = {}

    def run_first_request():
        first_result["value"] = service.infer({"video_path": str(video)})

    thread = threading.Thread(target=run_first_request)
    thread.start()

    assert started.wait(timeout=2), "first inference did not start"

    with pytest.raises(infer_server.APIError) as exc_info:
        service.infer({"video_path": str(video)})

    assert exc_info.value.status == 429
    assert exc_info.value.code == "server_busy"

    release.set()
    thread.join(timeout=5)
    service.shutdown()

    assert first_result["value"]["shape"] == [1, 1]


def test_health_payload_reports_readiness():
    service = infer_server.InferenceService(
        config=infer_server.ServerConfig(),
        inference_fn=lambda _video_path: {"ok": True},
    )

    unready = service.health_payload()
    assert unready["ready"] is False
    assert unready["ok"] is True

    service._ready = True
    ready = service.health_payload()
    assert ready["ready"] is True

    service.shutdown()
