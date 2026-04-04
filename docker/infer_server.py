#!/usr/bin/env python3
"""Persistent local inference server for hot Raramuri runs.

Usage:
    python infer_server.py --port 8765

POST /infer
{
  "video_path": "/workspace/clip.mp4",
    "video_url": "https://www.youtube.com/watch?v=...",
    "start_time": "00:00:00",
    "end_time": "00:00:30",
  "output": "/workspace/results.json"   # optional
}

POST /jobs/{id}/visualize
{
  "output": "/workspace/composite.mp4",  # optional
  "fps": 24,                              # optional, default 24
  "width": 1920,                          # optional, default 1920
  "height": 540                            # optional, default 540
}

GET /health
GET /ready
GET /status
GET /metrics
"""

import argparse
import concurrent.futures
import inspect
import json
import logging
import os
import threading
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from infer import (
    _get_parakeet_model,
    ensure_runtime_prerequisites,
    load_model,
    log_runtime_contract,
    patch_runtime_extractors,
    normalize_time_range,
    prepare_video_input,
    run_inference,
    runtime_status_snapshot,
    warm_runtime_model_dependencies,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Expected request or service error with an HTTP status code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ServerConfig:
    request_timeout_seconds: int = 900
    max_request_bytes: int = 64 * 1024
    max_pending_requests: int = 1

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            request_timeout_seconds=max(1, int(os.environ.get("RARAMURI_SERVER_REQUEST_TIMEOUT", "900"))),
            max_request_bytes=max(1024, int(os.environ.get("RARAMURI_SERVER_MAX_REQUEST_BYTES", str(64 * 1024)))),
            max_pending_requests=max(1, int(os.environ.get("RARAMURI_SERVER_MAX_PENDING_REQUESTS", "1"))),
        )


@dataclass
class JobRecord:
    job_id: str
    request: dict
    status: str
    submitted_at: str
    updated_at: str
    phase: str = "queued"
    stage: str = "accepted"
    progress_ratio: float = 0.0
    step: str = "queued"
    step_index: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: dict | None = None
    output_path: str | None = None
    result_shape: list | None = None
    result: dict | None = None
    video_file: str | None = None
    cancel_requested: bool = False
    future: concurrent.futures.Future | None = field(default=None, repr=False)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class InferenceService:
    """Serializes GPU inference behind a small, explicit service boundary."""

    def __init__(
        self,
        config: ServerConfig,
        inference_fn: Callable[[str], dict] = run_inference,
    ):
        self.config = config
        self._inference_fn = inference_fn
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="inference",
        )
        self._pending_slots = threading.BoundedSemaphore(config.max_pending_requests)
        self._lock = threading.Lock()
        self._active_requests = 0
        self._pending_requests = 0
        self._ready = False
        self._warming = False
        self._warm_error = None
        self._started_at = time.monotonic()
        self._request_sequence = 0
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._requests_rejected = 0
        self._current_request = None
        self._last_request = None
        self._jobs: dict[str, JobRecord] = {}
        self._supports_progress_callback = self._detect_progress_callback_support(inference_fn)

    def warm_models(self) -> None:
        """Prepare runtime patches and warm the cached models once at startup."""
        with self._lock:
            self._warming = True
            self._warm_error = None

        try:
            patch_runtime_extractors()

            t0 = time.monotonic()
            load_model(reuse=True)
            logger.info("TRIBEv2 model warmed in %.1fs", time.monotonic() - t0)

            t0 = time.monotonic()
            warm_runtime_model_dependencies()
            logger.info("HF model dependencies warmed in %.1fs", time.monotonic() - t0)

            transcript_backend = os.environ.get("RARAMURI_TRANSCRIPT_BACKEND", "").strip().lower()
            if transcript_backend == "parakeet":
                model_name = os.environ.get("RARAMURI_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
                t0 = time.monotonic()
                _get_parakeet_model(model_name)
                logger.info("Parakeet model warmed in %.1fs", time.monotonic() - t0)

            with self._lock:
                self._ready = True
        except Exception as exc:
            with self._lock:
                self._ready = False
                self._warm_error = str(exc)
            raise
        finally:
            with self._lock:
                self._warming = False

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def health_payload(self) -> dict:
        with self._lock:
            ready = self._ready
            warming = self._warming
            active = self._active_requests
            pending = self._pending_requests
            warm_error = self._warm_error
            requests_started = self._requests_started
            requests_completed = self._requests_completed
            requests_failed = self._requests_failed
            requests_rejected = self._requests_rejected

        return {
            "ok": True,
            "ready": ready,
            "warming": warming,
            "active_requests": active,
            "pending_requests": pending,
            "max_pending_requests": self.config.max_pending_requests,
            "requests_started": requests_started,
            "requests_completed": requests_completed,
            "requests_failed": requests_failed,
            "requests_rejected": requests_rejected,
            "uptime_seconds": round(time.monotonic() - self._started_at, 3),
            "warm_error": warm_error,
        }

    def readiness_status(self) -> tuple[int, dict]:
        payload = self.health_payload()
        return (200 if payload["ready"] else 503), payload

    def status_payload(self) -> dict:
        payload = self.health_payload()
        with self._lock:
            current_request = None if self._current_request is None else dict(self._current_request)
            last_request = None if self._last_request is None else dict(self._last_request)

        if current_request is not None and "started_monotonic" in current_request:
            current_request["elapsed_seconds"] = round(time.monotonic() - current_request["started_monotonic"], 3)
            current_request.pop("started_monotonic", None)
        if last_request is not None:
            last_request.pop("started_monotonic", None)

        payload["runtime"] = self._safe_runtime_snapshot()
        payload["current_request"] = current_request
        payload["last_request"] = last_request
        payload["jobs"] = {
            "total": len(self._jobs),
            "active": sum(1 for job in self._jobs.values() if job.status in {"queued", "running"}),
        }
        return payload

    def metrics_payload(self) -> str:
        status = self.status_payload()
        runtime = status.get("runtime") or {}
        process = runtime.get("process") or {}
        gpu = runtime.get("gpu") or {}
        current_request = status.get("current_request") or {}
        last_request = status.get("last_request") or {}

        lines = [
            "# HELP raramuri_server_ready Whether the inference server is ready to accept requests.",
            "# TYPE raramuri_server_ready gauge",
            f"raramuri_server_ready {1 if status['ready'] else 0}",
            "# HELP raramuri_server_warming Whether the inference server is currently warming models.",
            "# TYPE raramuri_server_warming gauge",
            f"raramuri_server_warming {1 if status['warming'] else 0}",
            "# HELP raramuri_server_active_requests Number of active inference requests.",
            "# TYPE raramuri_server_active_requests gauge",
            f"raramuri_server_active_requests {status['active_requests']}",
            "# HELP raramuri_server_pending_requests Number of admitted requests that are pending or running.",
            "# TYPE raramuri_server_pending_requests gauge",
            f"raramuri_server_pending_requests {status['pending_requests']}",
            "# HELP raramuri_server_requests_started_total Total admitted inference requests.",
            "# TYPE raramuri_server_requests_started_total counter",
            f"raramuri_server_requests_started_total {status['requests_started']}",
            "# HELP raramuri_server_requests_completed_total Total completed inference requests.",
            "# TYPE raramuri_server_requests_completed_total counter",
            f"raramuri_server_requests_completed_total {status['requests_completed']}",
            "# HELP raramuri_server_requests_failed_total Total failed inference requests.",
            "# TYPE raramuri_server_requests_failed_total counter",
            f"raramuri_server_requests_failed_total {status['requests_failed']}",
            "# HELP raramuri_server_requests_rejected_total Total rejected inference requests.",
            "# TYPE raramuri_server_requests_rejected_total counter",
            f"raramuri_server_requests_rejected_total {status['requests_rejected']}",
            "# HELP raramuri_server_uptime_seconds Process uptime in seconds.",
            "# TYPE raramuri_server_uptime_seconds gauge",
            f"raramuri_server_uptime_seconds {status['uptime_seconds']}",
            "# HELP raramuri_server_current_request_progress_ratio Fraction of the current request completed.",
            "# TYPE raramuri_server_current_request_progress_ratio gauge",
            f"raramuri_server_current_request_progress_ratio {current_request.get('progress_ratio', 0.0)}",
            "# HELP raramuri_server_current_request_elapsed_seconds Elapsed seconds for the current request.",
            "# TYPE raramuri_server_current_request_elapsed_seconds gauge",
            f"raramuri_server_current_request_elapsed_seconds {current_request.get('elapsed_seconds', 0.0)}",
            "# HELP raramuri_server_current_request_step_index Index of the current request step.",
            "# TYPE raramuri_server_current_request_step_index gauge",
            f"raramuri_server_current_request_step_index {current_request.get('step_index', 0)}",
            "# HELP raramuri_server_last_request_duration_seconds Duration of the most recent request.",
            "# TYPE raramuri_server_last_request_duration_seconds gauge",
            f"raramuri_server_last_request_duration_seconds {last_request.get('duration_seconds', 0.0)}",
        ]

        numeric_process_metrics = {
            "raramuri_server_process_rss_max_mb": process.get("rss_max_mb"),
            "raramuri_server_process_rss_current_mb": process.get("rss_current_mb"),
            "raramuri_server_process_vmsize_mb": process.get("vmsize_mb"),
            "raramuri_server_gpu_allocated_mb": gpu.get("allocated_mb"),
            "raramuri_server_gpu_reserved_mb": gpu.get("reserved_mb"),
            "raramuri_server_gpu_max_allocated_mb": gpu.get("max_allocated_mb"),
            "raramuri_server_gpu_max_reserved_mb": gpu.get("max_reserved_mb"),
        }
        for metric_name, metric_value in numeric_process_metrics.items():
            if metric_value is not None:
                lines.append(f"{metric_name} {metric_value}")

        return "\n".join(lines) + "\n"

    def infer(self, payload: dict) -> dict:
        if not self.health_payload()["ready"]:
            self._record_rejection()
            raise APIError(503, "service_unready", "Inference service is still warming")

        request = self._validate_payload(payload)

        if not self._pending_slots.acquire(blocking=False):
            self._record_rejection()
            raise APIError(429, "server_busy", "Inference capacity is saturated; retry later")

        with self._lock:
            self._request_sequence += 1
            request_id = self._request_sequence
            self._pending_requests += 1
            self._requests_started += 1

        future = self._executor.submit(self._run_inference_job, request_id, request)
        future.add_done_callback(self._release_slot)

        try:
            return future.result(timeout=self.config.request_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise APIError(504, "inference_timeout", "Inference request exceeded the server timeout") from exc
        except APIError:
            raise
        except Exception as exc:
            raise APIError(500, "inference_failed", str(exc)) from exc

    def submit_job(self, payload: dict) -> dict:
        if not self.health_payload()["ready"]:
            self._record_rejection()
            raise APIError(503, "service_unready", "Inference service is still warming")

        request = self._validate_payload(payload)

        if not self._pending_slots.acquire(blocking=False):
            self._record_rejection()
            raise APIError(429, "server_busy", "Inference capacity is saturated; retry later")

        with self._lock:
            self._request_sequence += 1
            self._pending_requests += 1
            self._requests_started += 1
            job_id = f"job_{uuid.uuid4().hex[:16]}"
            output_path = request.get("output")
            job = JobRecord(
                job_id=job_id,
                request=request,
                status="queued",
                submitted_at=_utc_now_iso(),
                updated_at=_utc_now_iso(),
                output_path=output_path,
            )
            self._jobs[job_id] = job

        future = self._executor.submit(self._run_job, job_id)
        with self._lock:
            self._jobs[job_id].future = future
        future.add_done_callback(self._release_slot)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            current_request = None if self._current_request is None else dict(self._current_request)
            if job is None:
                raise APIError(404, "job_not_found", f"Job not found: {job_id}")
            payload = {
                "job_id": job.job_id,
                "status": job.status,
                "phase": job.phase,
                "stage": job.stage,
                "progress_ratio": job.progress_ratio,
                "step": job.step,
                "step_index": job.step_index,
                "total_steps": job.total_steps,
                "completed_steps": job.completed_steps,
                "submitted_at": job.submitted_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "updated_at": job.updated_at,
                "cancel_requested": job.cancel_requested,
                "input": {k: v for k, v in job.request.items() if k != "output"},
                "output_path": job.output_path,
                "result_shape": job.result_shape,
                "error": job.error,
            }
            if current_request is not None and current_request.get("job_id") == job_id:
                payload["runtime"] = current_request.get("runtime")
                payload["elapsed_seconds"] = current_request.get("elapsed_seconds")
            else:
                payload["runtime"] = self._safe_runtime_snapshot()
        return payload

    def get_job_result(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "job_not_found", f"Job not found: {job_id}")
            if job.status == "completed" and job.result is not None:
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "output_path": job.output_path,
                    "result": job.result,
                }
            if job.status == "failed":
                raise APIError(409, "job_failed", "Job failed; inspect job status for details")
            if job.status == "cancelled":
                raise APIError(409, "job_cancelled", "Job was cancelled before completion")
        raise APIError(409, "job_not_ready", "Job is not complete yet")

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "job_not_found", f"Job not found: {job_id}")
            if job.status in {"completed", "failed", "cancelled"}:
                raise APIError(409, "job_not_cancellable", f"Job is already {job.status}")
            job.cancel_requested = True
            job.updated_at = _utc_now_iso()
            future = job.future

        if future is not None and future.cancel():
            with self._lock:
                job.status = "cancelled"
                job.phase = "cancelled"
                job.stage = "completed"
                job.completed_at = _utc_now_iso()
                job.updated_at = job.completed_at
            return self.get_job(job_id)

        raise APIError(409, "job_not_cancellable", "Job is already running and cannot be cancelled")

    def render_visualization(self, job_id: str, options: dict) -> dict:
        """Render a composite visualization video for a completed job.

        The render runs synchronously on a background thread (same executor).
        Returns render metadata including output_path and timing.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise APIError(404, "job_not_found", f"Job not found: {job_id}")
            if job.status != "completed" or job.result is None:
                raise APIError(409, "job_not_ready", "Job must be completed before visualization")
            result = job.result
            # Recover the video source from the original request
            video_source = job.request.get("video_path") or job.request.get("video_url")

        fps = max(1, min(60, int(options.get("fps", 24))))
        width = max(640, min(3840, int(options.get("width", 1920))))
        height = max(270, min(2160, int(options.get("height", 540))))
        output_path = options.get("output")

        if output_path is None:
            output_dir = Path(tempfile.mkdtemp(prefix="raramuri-viz-out-"))
            output_path = str(output_dir / f"{job_id}_composite.mp4")

        # We need the local video file. If it was a remote URL, re-download.
        cleanup_dir = None
        if video_source and "://" in video_source:
            cleanup_dir = Path(tempfile.mkdtemp(prefix="raramuri-viz-src-"))
            local_video = str(cleanup_dir / "source.mp4")
            from infer import prepare_video_input
            prepared_path, _, _ = prepare_video_input(
                video_source,
                start_time=job.request.get("start_time"),
                end_time=job.request.get("end_time"),
            )
            local_video = str(prepared_path)
        else:
            local_video = video_source

        if not local_video or not Path(local_video).exists():
            raise APIError(400, "video_unavailable", "Source video is no longer available for visualization")

        try:
            from render_viz import render_composite_video
            render_result = render_composite_video(
                result=result,
                video_path=local_video,
                output_path=output_path,
                fps=fps,
                width_px=width,
                height_px=height,
            )
            return render_result
        finally:
            if cleanup_dir is not None:
                import shutil as _shutil
                _shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _run_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            request = dict(job.request)
        request_id = int(time.time() * 1000)
        return self._run_inference_job(request_id, request, job_id=job_id)

    def _run_inference_job(self, request_id: int, request: dict, job_id: str | None = None) -> dict:
        input_source = request.get("video_path") or request.get("video_url")
        output_path = request.get("output")
        cleanup_dir = None
        started_at = time.monotonic()
        with self._lock:
            self._active_requests += 1
            self._current_request = {
                "request_id": request_id,
                "job_id": job_id,
                "video_path": input_source,
                "status": "running",
                "step": "queued",
                "stage": "started",
                "step_index": 0,
                "total_steps": 0,
                "completed_steps": 0,
                "progress_ratio": 0.0,
                "elapsed_seconds": 0.0,
                "started_monotonic": started_at,
                "runtime": self._safe_runtime_snapshot(),
            }
            if job_id is not None:
                job = self._jobs[job_id]
                job.status = "running"
                job.phase = "preparing"
                job.stage = "started"
                job.started_at = _utc_now_iso()
                job.updated_at = job.started_at
        try:
            prepared_video_path, transfer_metadata, cleanup_dir = prepare_video_input(
                input_source,
                start_time=request.get("start_time"),
                end_time=request.get("end_time"),
            )
            progress_callback = self._make_progress_callback(request_id, input_source, started_at, job_id=job_id)
            result = self._invoke_inference_fn(str(prepared_video_path), progress_callback)
            result.setdefault("timing", {})["transfer"] = {
                **result.get("timing", {}).get("transfer", {}),
                "input_prepare": transfer_metadata,
            }
            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result))
            duration = round(time.monotonic() - started_at, 3)
            with self._lock:
                self._requests_completed += 1
                self._last_request = {
                    "request_id": request_id,
                    "job_id": job_id,
                    "video_path": input_source,
                    "status": "completed",
                    "duration_seconds": duration,
                    "finished_step": self._current_request.get("step") if self._current_request else None,
                    "result_shape": result.get("shape"),
                    "runtime": self._safe_runtime_snapshot(),
                }
                if job_id is not None:
                    job = self._jobs[job_id]
                    job.status = "completed"
                    job.phase = "completed"
                    job.stage = "completed"
                    job.result = result
                    job.result_shape = result.get("shape")
                    job.completed_at = _utc_now_iso()
                    job.updated_at = job.completed_at
                    # Preserve video file for client download
                    if cleanup_dir is not None and prepared_video_path.exists():
                        video_dest = Path(tempfile.mkdtemp(prefix="raramuri-video-")) / f"{job_id}.mp4"
                        import shutil as _shutil2
                        _shutil2.copy2(str(prepared_video_path), str(video_dest))
                        job.video_file = str(video_dest)
                self._current_request = None
            return result
        except Exception as exc:
            duration = round(time.monotonic() - started_at, 3)
            with self._lock:
                self._requests_failed += 1
                current_request = dict(self._current_request) if self._current_request is not None else {}
                self._last_request = {
                    "request_id": request_id,
                    "job_id": job_id,
                    "video_path": input_source,
                    "status": "failed",
                    "duration_seconds": duration,
                    "finished_step": current_request.get("step"),
                    "runtime": self._safe_runtime_snapshot(),
                }
                if job_id is not None:
                    job = self._jobs[job_id]
                    job.status = "failed"
                    job.phase = current_request.get("step") or job.phase
                    job.stage = "completed"
                    job.completed_at = _utc_now_iso()
                    job.updated_at = job.completed_at
                    job.error = {
                        "message": str(exc),
                        "code": "job_failed",
                    }
                self._current_request = None
            raise
        finally:
            if cleanup_dir is not None:
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            with self._lock:
                self._active_requests -= 1

    def _release_slot(self, _future: concurrent.futures.Future) -> None:
        with self._lock:
            self._pending_requests -= 1
        self._pending_slots.release()

    def _make_progress_callback(self, request_id: int, video_path: str, started_at: float, job_id: str | None = None) -> Callable[[dict], None]:
        def _callback(update: dict) -> None:
            with self._lock:
                if self._current_request is None or self._current_request.get("request_id") != request_id:
                    return
                self._current_request.update(
                    {
                        "video_path": video_path,
                        "status": "running",
                        "step": update.get("step", self._current_request.get("step")),
                        "stage": update.get("stage", self._current_request.get("stage")),
                        "step_index": update.get("step_index", self._current_request.get("step_index", 0)),
                        "total_steps": update.get("total_steps", self._current_request.get("total_steps", 0)),
                        "completed_steps": update.get("completed_steps", self._current_request.get("completed_steps", 0)),
                        "progress_ratio": update.get("progress_ratio", self._current_request.get("progress_ratio", 0.0)),
                        "elapsed_seconds": update.get("elapsed_seconds", round(time.monotonic() - started_at, 3)),
                        "step_elapsed_seconds": update.get("step_elapsed_seconds"),
                        "runtime": update.get("runtime", self._safe_runtime_snapshot()),
                    }
                )
                if job_id is not None and job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = "running"
                    job.phase = update.get("step", job.phase)
                    job.stage = update.get("stage", job.stage)
                    job.progress_ratio = update.get("progress_ratio", job.progress_ratio)
                    job.step = update.get("step", job.step)
                    job.step_index = update.get("step_index", job.step_index)
                    job.total_steps = update.get("total_steps", job.total_steps)
                    job.completed_steps = update.get("completed_steps", job.completed_steps)
                    job.updated_at = _utc_now_iso()

        return _callback

    def _invoke_inference_fn(self, video_path: str, progress_callback: Callable[[dict], None]) -> dict:
        if self._supports_progress_callback:
            return self._inference_fn(video_path, progress_callback=progress_callback)
        return self._inference_fn(video_path)

    @staticmethod
    def _detect_progress_callback_support(inference_fn: Callable) -> bool:
        try:
            signature = inspect.signature(inference_fn)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return "progress_callback" in signature.parameters

    def _record_rejection(self) -> None:
        with self._lock:
            self._requests_rejected += 1

    @staticmethod
    def _safe_runtime_snapshot() -> dict:
        try:
            return runtime_status_snapshot()
        except Exception:
            return {"process": {}, "gpu": None}

    @staticmethod
    def _validate_payload(payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise APIError(400, "invalid_payload", "Request body must be a JSON object")

        allowed_fields = {"video_path", "video_url", "start_time", "end_time", "output"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise APIError(400, "unknown_fields", f"Unsupported request fields: {', '.join(unknown_fields)}")

        video_path = payload.get("video_path")
        video_url = payload.get("video_url")

        if bool(video_path) == bool(video_url):
            raise APIError(400, "invalid_video_source", "Provide exactly one of video_path or video_url")

        cleaned = {}
        if video_path is not None:
            if not isinstance(video_path, str) or not video_path.strip():
                raise APIError(400, "missing_video_path", "video_path must be a non-empty string")
            if "://" in video_path:
                raise APIError(400, "remote_video_unsupported", "Use video_url for remote sources")

            video = Path(video_path).expanduser()
            if not video.exists() or not video.is_file():
                raise APIError(400, "video_not_found", f"video_path does not exist: {video}")
            cleaned["video_path"] = str(video)
        else:
            if not isinstance(video_url, str) or not video_url.strip():
                raise APIError(400, "missing_video_url", "video_url must be a non-empty string")
            if "://" not in video_url:
                raise APIError(400, "invalid_video_url", "video_url must be an absolute URL")
            cleaned["video_url"] = video_url.strip()

        try:
            start_time, end_time = normalize_time_range(payload.get("start_time"), payload.get("end_time"))
        except ValueError as exc:
            raise APIError(400, "invalid_time_range", str(exc)) from exc

        if start_time is not None:
            cleaned["start_time"] = start_time
        if end_time is not None:
            cleaned["end_time"] = end_time

        output_path = payload.get("output")
        if output_path is not None:
            if not isinstance(output_path, str) or not output_path.strip():
                raise APIError(400, "invalid_output_path", "output must be a non-empty string when provided")
            output = Path(output_path).expanduser()
            if output.exists() and output.is_dir():
                raise APIError(400, "invalid_output_path", "output must be a file path, not a directory")
            cleaned["output"] = str(Path(output_path).expanduser())

        return cleaned


class InferenceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, service: InferenceService):
        super().__init__(server_address, handler_class)
        self.service = service


class Handler(BaseHTTPRequestHandler):
    server_version = "RaramuriInfer/0.4"

    @property
    def service(self) -> InferenceService:
        return self.server.service

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str):
        self._send_json(status, {"error": message, "code": code})

    def _send_text(self, status: int, body: str, content_type: str = "text/plain; version=0.0.4"):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, file_path: str, content_type: str = "video/mp4"):
        fpath = Path(file_path)
        if not fpath.exists() or not fpath.is_file():
            self._send_error(404, "file_not_found", "File not found")
            return
        file_size = fpath.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        if content_type.startswith("video/"):
            self.send_header("Content-Disposition", f'inline; filename="{fpath.name}"')
        else:
            self.send_header("Content-Disposition", f'attachment; filename="{fpath.name}"')
        self.end_headers()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def _read_json_payload(self) -> dict:
        if self.headers.get("Content-Type", "application/json").split(";")[0].strip().lower() != "application/json":
            raise APIError(415, "unsupported_media_type", "Content-Type must be application/json")

        content_length_header = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_header)
        except ValueError as exc:
            raise APIError(400, "invalid_content_length", "Content-Length must be an integer") from exc

        if content_length <= 0:
            raise APIError(400, "empty_body", "Request body is required")
        if content_length > self.service.config.max_request_bytes:
            raise APIError(413, "request_too_large", "Request body exceeds the configured limit")

        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIError(400, "invalid_json", "Request body must be valid JSON") from exc

    def log_message(self, format: str, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    @staticmethod
    def _job_urls(job_id: str) -> dict:
        return {
            "status_url": f"/jobs/{job_id}",
            "result_url": f"/jobs/{job_id}/result",
            "cancel_url": f"/jobs/{job_id}/cancel",
        }

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, self.service.health_payload())
            return
        if path == "/ready":
            status, payload = self.service.readiness_status()
            self._send_json(status, payload)
            return
        if path == "/status":
            self._send_json(200, self.service.status_payload())
            return
        if path == "/metrics":
            self._send_text(200, self.service.metrics_payload())
            return
        if path == "/jobs":
            with self.service._lock:
                jobs = []
                for jid, jr in self.service._jobs.items():
                    jobs.append({
                        "job_id": jr.job_id,
                        "status": jr.status,
                        "phase": jr.phase,
                        "submitted_at": jr.submitted_at,
                        "completed_at": jr.completed_at,
                        "result_shape": jr.result_shape,
                        "has_video": jr.video_file is not None and Path(jr.video_file).exists(),
                        "input": {k: v for k, v in jr.request.items() if k != "output"},
                    })
            self._send_json(200, {"jobs": jobs})
            return
        if path.startswith("/jobs/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 2 and parts[0] == "jobs":
                try:
                    payload = self.service.get_job(parts[1])
                    payload.update(self._job_urls(parts[1]))
                    self._send_json(200, payload)
                    return
                except APIError as exc:
                    self._send_error(exc.status, exc.code, exc.message)
                    return
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "result":
                try:
                    payload = self.service.get_job_result(parts[1])
                    payload.update(self._job_urls(parts[1]))
                    self._send_json(200, payload)
                    return
                except APIError as exc:
                    self._send_error(exc.status, exc.code, exc.message)
                    return
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "video":
                try:
                    job = self.service.get_job(parts[1])
                    video_file = job.get("video_file") if isinstance(job, dict) else None
                    if video_file is None:
                        with self.service._lock:
                            jr = self.service._jobs.get(parts[1])
                            video_file = jr.video_file if jr else None
                    if video_file and Path(video_file).exists():
                        self._send_file(video_file, content_type="video/mp4")
                        return
                    self._send_error(404, "video_not_found", "Video file not available for this job")
                    return
                except APIError as exc:
                    self._send_error(exc.status, exc.code, exc.message)
                    return
        if path.startswith("/files/"):
            filename = path.split("/files/", 1)[1]
            if not filename or "/" in filename or filename.startswith("."):
                self._send_error(400, "invalid_filename", "Invalid filename")
                return
            # Search tmp dirs for the file
            import glob
            matches = glob.glob(f"/tmp/raramuri-viz-out-*/{filename}")
            if matches:
                self._send_file(matches[0])
                return
            self._send_error(404, "file_not_found", "File not found")
            return
        self._send_error(404, "not_found", "not found")

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            payload = self._read_json_payload()
            if path == "/infer":
                result = self.service.infer(payload)
                self._send_json(200, result)
                return
            if path == "/jobs":
                result = self.service.submit_job(payload)
                result.update(self._job_urls(result["job_id"]))
                self._send_json(202, result)
                return
            if path.startswith("/jobs/") and path.endswith("/cancel"):
                parts = [part for part in path.split("/") if part]
                if len(parts) == 3 and parts[0] == "jobs":
                    result = self.service.cancel_job(parts[1])
                    result.update(self._job_urls(parts[1]))
                    self._send_json(200, result)
                    return
            if path.startswith("/jobs/") and path.endswith("/visualize"):
                parts = [part for part in path.split("/") if part]
                if len(parts) == 3 and parts[0] == "jobs":
                    result = self.service.render_visualization(parts[1], payload)
                    result["download_url"] = f"/files/{Path(result['output_path']).name}"
                    self._send_json(200, result)
                    return
            self._send_error(404, "not_found", "not found")
        except APIError as exc:
            self._send_error(exc.status, exc.code, exc.message)
        except Exception as exc:
            logger.exception("infer request failed")
            self._send_error(500, "internal_error", str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ensure_runtime_prerequisites()
    log_runtime_contract()

    config = ServerConfig.from_env()
    service = InferenceService(config=config)
    service.warm_models()

    server = InferenceHTTPServer((args.host, args.port), Handler, service)
    logger.info(
        "Persistent inference server listening on %s:%d (max_pending_requests=%d timeout=%ds)",
        args.host,
        args.port,
        config.max_pending_requests,
        config.request_timeout_seconds,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.shutdown()


if __name__ == "__main__":
    main()
