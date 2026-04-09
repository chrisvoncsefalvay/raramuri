"""RunPod handler for Rarámuri inference — supports serverless AND pod mode.

Serverless mode (default): RunPod dispatches jobs via runpod.serverless.start().
Pod mode (RARAMURI_POD_MODE=1): Starts a simple HTTP server for interactive testing.

Module-level warm-up loads all models once per worker lifetime.
Each handler invocation runs inference and returns phase-wise timing
in Prometheus-compatible format alongside the full result bundle.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────

_warmup_timings = {}
_warmup_total_seconds = 0.0
_models_loaded = False
_requests_served = 0

# ── Network-volume model discovery ──────────────────────────────────

VOLUME_SEARCH_PATHS = ("/runpod-volume", "/workspace")


def _restore_symlinks(models_root: Path):
    """Restore HF cache symlinks that were uploaded as text files via S3.

    The S3 upload script stores symlinks as small text files whose content is
    the relative link target (e.g. "../../blobs/abc123").  This function finds
    those files in snapshots/ dirs and converts them back to real symlinks so
    that transformers' cache resolution works correctly.
    """
    restored = 0
    for snapshots_dir in models_root.rglob("snapshots"):
        if not snapshots_dir.is_dir():
            continue
        for f in snapshots_dir.rglob("*"):
            if not f.is_file() or f.is_symlink() or f.stat().st_size > 200:
                continue
            content = f.read_text().strip()
            if content.startswith("../../blobs/") or content.startswith("../../../blobs/"):
                target = f.parent / content
                if target.exists():
                    f.unlink()
                    f.symlink_to(content)
                    restored += 1
    if restored:
        logger.info("Restored %d symlinks in %s", restored, models_root)


def _discover_volume_models():
    """Find models on a RunPod network volume and symlink cache dirs.

    RunPod mounts network volumes at /runpod-volume (serverless) or
    /workspace (pods).  preload_to_volume.py writes models into
    <mount>/models/{hf,tribe,mne_data} with a .ready marker.

    If a volume is found, we point HF_HOME / TRIBE_CACHE / MNE_DATA
    at the volume paths so infer.py picks them up without downloading.
    """
    for vol_root in VOLUME_SEARCH_PATHS:
        marker = Path(vol_root) / "models" / ".ready"
        if marker.is_file():
            models_root = Path(vol_root) / "models"
            logger.info("Found pre-staged models at %s", models_root)

            _restore_symlinks(models_root)

            mappings = {
                "HF_HOME": str(models_root / "hf"),
                "TRIBE_CACHE": str(models_root / "tribe"),
                "MNE_DATA": str(models_root / "mne_data"),
            }
            for env_var, path in mappings.items():
                if Path(path).is_dir():
                    os.environ[env_var] = path
                    logger.info("  %s -> %s", env_var, path)
            return True

    logger.info("No pre-staged volume models found; will download on first use")
    return False


# ── Lazy model loading ──────────────────────────────────────────────

from infer import (
    _get_parakeet_model,
    ensure_runtime_prerequisites,
    load_model,
    log_runtime_contract,
    patch_runtime_extractors,
    prepare_video_input,
    run_inference,
    runtime_status_snapshot,
    warm_runtime_model_dependencies,
    normalize_time_range,
)


def _lazy_init():
    """Load all models on first request, not at module level.

    RunPod's serverless health check has a tight init timeout. By deferring
    heavy model loading to the first handler invocation, the worker reports
    ready immediately and the models load while processing the first job.
    """
    global _models_loaded, _warmup_timings, _warmup_total_seconds
    if _models_loaded:
        return

    wall_start = time.monotonic()

    _discover_volume_models()

    log_runtime_contract()
    ensure_runtime_prerequisites()
    patch_runtime_extractors()

    t0 = time.monotonic()
    load_model(reuse=True)
    _warmup_timings["tribev2"] = round(time.monotonic() - t0, 3)
    logger.info("TRIBEv2 model warmed in %.1fs", _warmup_timings["tribev2"])

    t0 = time.monotonic()
    warm_runtime_model_dependencies()
    _warmup_timings["hf_dependencies"] = round(time.monotonic() - t0, 3)
    logger.info("HF model dependencies warmed in %.1fs", _warmup_timings["hf_dependencies"])

    transcript_backend = os.environ.get("RARAMURI_TRANSCRIPT_BACKEND", "").strip().lower()
    if transcript_backend == "parakeet":
        model_name = os.environ.get("RARAMURI_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
        t0 = time.monotonic()
        _get_parakeet_model(model_name)
        _warmup_timings["parakeet"] = round(time.monotonic() - t0, 3)
        logger.info("Parakeet model warmed in %.1fs", _warmup_timings["parakeet"])

    _warmup_total_seconds = round(time.monotonic() - wall_start, 3)
    logger.info("RunPod worker warm-up complete in %.1fs: %s", _warmup_total_seconds, _warmup_timings)
    _models_loaded = True


# ── Prometheus metrics ──────────────────────────────────────────────

def _format_prometheus_metrics(
    phase_timings: dict,
    warmup_timings: dict,
    warmup_total: float,
    total_seconds: float,
    is_warm_start: bool,
    gpu_snapshot: dict | None,
) -> str:
    """Render phase-wise timing as Prometheus text exposition format."""
    lines = []

    lines.append("# HELP raramuri_warmup_total_seconds Total wall-clock time for model warm-up.")
    lines.append("# TYPE raramuri_warmup_total_seconds gauge")
    lines.append(f"raramuri_warmup_total_seconds {warmup_total}")

    lines.append("# HELP raramuri_warmup_phase_seconds Duration of each warm-up phase.")
    lines.append("# TYPE raramuri_warmup_phase_seconds gauge")
    for phase, secs in warmup_timings.items():
        lines.append(f'raramuri_warmup_phase_seconds{{phase="{phase}"}} {secs}')

    lines.append("# HELP raramuri_inference_total_seconds Total inference wall-clock time.")
    lines.append("# TYPE raramuri_inference_total_seconds gauge")
    lines.append(f"raramuri_inference_total_seconds {total_seconds}")

    lines.append("# HELP raramuri_phase_seconds Duration of each inference phase.")
    lines.append("# TYPE raramuri_phase_seconds gauge")
    for phase, secs in phase_timings.items():
        if isinstance(secs, (int, float)):
            lines.append(f'raramuri_phase_seconds{{phase="{phase}"}} {secs}')

    lines.append("# HELP raramuri_warm_start Whether this was a warm-start inference (models already in GPU).")
    lines.append("# TYPE raramuri_warm_start gauge")
    lines.append(f"raramuri_warm_start {1 if is_warm_start else 0}")

    if gpu_snapshot:
        for key in ("allocated_mb", "reserved_mb", "max_allocated_mb", "max_reserved_mb"):
            val = gpu_snapshot.get(key)
            if val is not None:
                lines.append(f"raramuri_gpu_{key} {val}")

    return "\n".join(lines) + "\n"


# ── Helpers ────────────────────────────────────────────────────────

_CHUNK_SECONDS = int(os.environ.get("RARAMURI_CHUNK_SECONDS", "20"))


def _probe_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _split_chunk(video_path: str, start: float, duration: float, output_path: str):
    """Extract a chunk from a video with ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", video_path,
         "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-avoid_negative_ts", "1", output_path],
        capture_output=True, check=True,
    )


def _run_inference_threaded(video_path, progress_queue):
    """Run inference in a background thread, pushing progress to a queue."""
    def progress_callback(update):
        progress_queue.put({
            "type": "progress",
            "step": update.get("step"),
            "stage": update.get("stage"),
            "step_index": update.get("step_index"),
            "total_steps": update.get("total_steps"),
            "progress_ratio": update.get("progress_ratio"),
            "elapsed_seconds": update.get("elapsed_seconds"),
            "step_elapsed_seconds": update.get("step_elapsed_seconds"),
        })

    try:
        result = run_inference(video_path, progress_callback=progress_callback)
        progress_queue.put(("DONE", result))
    except Exception as exc:
        import traceback
        # Capture the full traceback in the thread — it's lost after re-raise
        # in the main thread.  Attach it as an attribute for the error handler.
        exc.__thread_traceback__ = traceback.format_exc()
        logger.exception("Inference failed in worker thread")
        progress_queue.put(("ERROR", exc))


def _drain_progress(progress_queue, timeout=0.1):
    """Drain all pending progress updates from the queue."""
    updates = []
    while True:
        try:
            item = progress_queue.get(timeout=timeout)
            if isinstance(item, tuple):
                return updates, item  # terminal signal
            updates.append(item)
        except queue.Empty:
            return updates, None


# ── Core inference handler ──────────────────────────────────────────

def handler(job):
    """RunPod serverless streaming handler with chunked output.

    Streams real-time progress updates via a background thread and emits
    results in chunks (default 20s) so clients receive partial output as
    each chunk completes.  RunPod aggregates all yielded values so that
    ``/run`` + ``/status`` still return everything, while ``/stream/{job_id}``
    lets clients consume updates incrementally.

    Input schema (job["input"]):
        video_url: str          -- YouTube URL or direct video URL
        video_path: str         -- path on network volume (alternative to URL)
        start_time: str | None  -- e.g. "00:00:10"
        end_time: str | None    -- e.g. "00:00:40"
        include_predictions: bool -- include raw predictions array (default True)
        chunk_seconds: int | None -- chunk size in seconds (default: env RARAMURI_CHUNK_SECONDS or 20)

    Yields:
        ``{"type": "progress", ...}`` — real-time phase updates during each chunk.
        ``{"type": "chunk_result", "chunk_index": 0, "chunk_range": [0, 20], ...}`` — per-chunk results.
        ``{"type": "result", ...}`` — final aggregated result (last yield).
    """
    global _requests_served

    _lazy_init()

    inp = job["input"]
    video_url = inp.get("video_url")
    video_path = inp.get("video_path")
    start_time = inp.get("start_time")
    end_time = inp.get("end_time")
    include_predictions = inp.get("include_predictions", True)
    chunk_seconds = inp.get("chunk_seconds", _CHUNK_SECONDS)

    if not video_url and not video_path:
        yield {"type": "error", "error": "Either video_url or video_path is required"}
        return

    is_warm_start = _requests_served > 0
    _requests_served += 1

    source = video_path if video_path else video_url
    cleanup_dir = None

    try:
        # Normalize time range.
        try:
            start_time, end_time = normalize_time_range(start_time, end_time)
        except ValueError as exc:
            yield {"type": "error", "error": f"Invalid time range: {exc}"}
            return

        # Prepare video (download if remote, trim if needed).
        prepared_path, transfer_metadata, cleanup_dir = prepare_video_input(
            source, start_time=start_time, end_time=end_time,
        )

        yield {"type": "progress", "step": "video_prepared", "stage": "completed",
               "transfer": transfer_metadata}

        # Determine chunking.
        duration = _probe_duration(str(prepared_path))
        if chunk_seconds and chunk_seconds > 0 and duration > chunk_seconds * 1.5:
            # Split into chunks — worth it only if video is >1.5x chunk size.
            chunks = []
            t = 0.0
            while t < duration:
                chunk_dur = min(chunk_seconds, duration - t)
                chunks.append((t, chunk_dur))
                t += chunk_dur
        else:
            # Single chunk — the whole video.
            chunks = [(0.0, duration)]

        total_chunks = len(chunks)
        yield {"type": "progress", "step": "chunking", "stage": "completed",
               "duration": round(duration, 2), "total_chunks": total_chunks,
               "chunk_seconds": chunk_seconds}

        wall_start = time.monotonic()
        all_chunk_results = []

        for chunk_idx, (chunk_start, chunk_dur) in enumerate(chunks):
            # Prepare chunk video file.
            if total_chunks > 1:
                import tempfile
                chunk_path = Path(tempfile.mktemp(
                    prefix=f"chunk{chunk_idx}_", suffix=".mp4",
                    dir=cleanup_dir or "/tmp",
                ))
                _split_chunk(str(prepared_path), chunk_start, chunk_dur, str(chunk_path))
                infer_path = str(chunk_path)
            else:
                infer_path = str(prepared_path)

            yield {"type": "progress", "step": "inference",
                   "stage": "started", "chunk_index": chunk_idx,
                   "chunk_range": [round(chunk_start, 2), round(chunk_start + chunk_dur, 2)],
                   "total_chunks": total_chunks}

            # Run inference in a background thread for real-time progress.
            progress_queue = queue.Queue()
            thread = threading.Thread(
                target=_run_inference_threaded,
                args=(infer_path, progress_queue),
                daemon=True,
            )
            thread.start()

            # Yield progress updates in real time as inference runs.
            terminal = None
            while terminal is None:
                updates, terminal = _drain_progress(progress_queue, timeout=1.0)
                for update in updates:
                    update["chunk_index"] = chunk_idx
                    update["total_chunks"] = total_chunks
                    yield update

            thread.join(timeout=5)

            if terminal[0] == "ERROR":
                raise terminal[1]

            chunk_result = terminal[1]

            # Offset captions/segments timestamps to absolute position.
            if chunk_start > 0:
                for caption in chunk_result.get("captions", []):
                    caption["start"] = round(caption["start"] + chunk_start, 3)
                    caption["end"] = round(caption["end"] + chunk_start, 3)
                for seg in chunk_result.get("segments", []):
                    seg["start"] = round(seg["start"] + chunk_start, 3)

            # Emit chunk result.
            chunk_output = {
                "type": "chunk_result",
                "chunk_index": chunk_idx,
                "chunk_range": [round(chunk_start, 2), round(chunk_start + chunk_dur, 2)],
                "total_chunks": total_chunks,
                "captions": chunk_result.get("captions", []),
                "event_types": chunk_result.get("event_types", {}),
                "has_text": chunk_result.get("has_text", False),
                "timing": chunk_result.get("timing", {}),
            }
            if include_predictions:
                chunk_output["predictions"] = chunk_result.get("predictions")
                chunk_output["spectrum"] = chunk_result.get("spectrum")
            # Include metrics only for single-chunk or last chunk.
            if total_chunks == 1 or chunk_idx == total_chunks - 1:
                chunk_output["metrics"] = chunk_result.get("metrics")

            yield chunk_output
            all_chunk_results.append(chunk_result)

        # Build final aggregated result.
        total_seconds = round(time.monotonic() - wall_start, 3)
        if total_chunks == 1:
            final = all_chunk_results[0]
        else:
            # Merge chunk results.
            final = {
                "captions": [],
                "event_types": {},
                "has_text": False,
                "timing": {"phases": {}, "total_seconds": total_seconds},
                "metrics": all_chunk_results[-1].get("metrics"),
            }
            if include_predictions:
                final["predictions"] = []
            for cr in all_chunk_results:
                final["captions"].extend(cr.get("captions", []))
                final["has_text"] = final["has_text"] or cr.get("has_text", False)
                if include_predictions and cr.get("predictions"):
                    final["predictions"].extend(cr["predictions"])
                for et, count in cr.get("event_types", {}).items():
                    final["event_types"][et] = final["event_types"].get(et, 0) + count

        # Enrich timing block.
        final.setdefault("timing", {})
        final["timing"]["total_seconds"] = total_seconds
        final["timing"]["warmup"] = {
            "total_seconds": _warmup_total_seconds,
            "phases": _warmup_timings,
        }
        final["timing"]["warm_start"] = is_warm_start
        final["timing"]["transfer"] = transfer_metadata
        final["timing"]["total_chunks"] = total_chunks
        final["timing"]["chunk_seconds"] = chunk_seconds

        # GPU snapshot.
        gpu_snapshot = None
        try:
            snap = runtime_status_snapshot()
            gpu_snapshot = snap.get("gpu")
        except Exception:
            pass

        phase_timings = final["timing"].get("phases", {})
        final["metrics_text"] = _format_prometheus_metrics(
            phase_timings=phase_timings,
            warmup_timings=_warmup_timings,
            warmup_total=_warmup_total_seconds,
            total_seconds=total_seconds,
            is_warm_start=is_warm_start,
            gpu_snapshot=gpu_snapshot,
        )

        if not include_predictions:
            final.pop("predictions", None)
            final.pop("spectrum", None)

        final["type"] = "result"
        yield final

    except Exception as exc:
        import traceback
        logger.exception("Inference failed")
        err_msg = f"{type(exc).__name__}: {exc}" if str(exc) else f"{type(exc).__name__}: {exc!r}"
        # Prefer thread-captured traceback (has the real stack) over re-raise tb.
        err_tb = getattr(exc, "__thread_traceback__", None) or traceback.format_exc()
        yield {"type": "error", "error": err_msg, "traceback": err_tb}
    finally:
        if cleanup_dir is not None:
            import shutil
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ── Pod-mode HTTP server ────────────────────────────────────────────

class _PodHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for pod-mode testing."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "models_loaded": _models_loaded,
                "requests_served": _requests_served,
            }).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Raramuri pod mode. POST /run with {\"input\": {...}} to run inference.\n")

    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            job = json.loads(body)
        except json.JSONDecodeError as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {exc}"}).encode())
            return

        logger.info("Pod-mode inference request: %s", {k: v for k, v in job.get("input", {}).items() if k != "include_predictions"})
        result = handler(job)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode())

    def log_message(self, format, *args):
        logger.info(format, *args)


def _run_pod_server():
    """Start an HTTP server for pod-mode testing on port 8888."""
    port = int(os.environ.get("RARAMURI_POD_PORT", "8888"))
    server = HTTPServer(("0.0.0.0", port), _PodHandler)
    logger.info("Pod mode: HTTP server on port %d (GET /health, POST /run)", port)
    logger.info("To run inference: curl -X POST http://localhost:%d/run -d '{\"input\": {\"video_url\": \"...\"}}'", port)
    server.serve_forever()


# ── Entrypoint ──────────────────────────────────────────────────────

def _is_pod_mode() -> bool:
    """Detect whether we're running in a RunPod pod (not serverless)."""
    if os.environ.get("RARAMURI_POD_MODE", "").strip() == "1":
        return True
    # RunPod serverless sets RUNPOD_ENDPOINT_ID; pods don't.
    if not os.environ.get("RUNPOD_ENDPOINT_ID") and os.environ.get("RUNPOD_POD_ID"):
        return True
    return False


if _is_pod_mode():
    logger.info("Detected pod mode — starting HTTP server instead of serverless handler")
    _run_pod_server()
else:
    import runpod
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })
