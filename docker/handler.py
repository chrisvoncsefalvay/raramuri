"""RunPod Serverless handler for Rarámuri inference.

Module-level warm-up loads all models once per worker lifetime.
Each handler invocation runs inference and returns phase-wise timing
in Prometheus-compatible format alongside the full result bundle.
"""

import logging
import os
import time

import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Module-level warm-up ──────────────────────────────────────────
# RunPod loads this module once per worker. Everything here runs at
# cold-start time and persists across requests on the same worker.

_warmup_timings = {}
_warmup_wall_start = time.monotonic()

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

_warmup_total_seconds = round(time.monotonic() - _warmup_wall_start, 3)
logger.info("RunPod worker warm-up complete in %.1fs: %s", _warmup_total_seconds, _warmup_timings)

# Track whether this worker has served a request (warm vs cold inference).
_requests_served = 0


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


def handler(job):
    """RunPod serverless handler.

    Input schema (job["input"]):
        video_url: str          -- YouTube URL or direct video URL
        video_path: str         -- path on network volume (alternative to URL)
        start_time: str | None  -- e.g. "00:00:10"
        end_time: str | None    -- e.g. "00:00:40"
        include_predictions: bool -- include raw predictions array (default True)

    Returns dict with:
        - timing.phases: per-phase seconds
        - timing.total_seconds: wall-clock total
        - timing.warmup: warm-up breakdown
        - timing.warm_start: bool
        - metrics_text: Prometheus text exposition
        - shape, segments, event_types, etc.
    """
    global _requests_served

    inp = job["input"]
    video_url = inp.get("video_url")
    video_path = inp.get("video_path")
    start_time = inp.get("start_time")
    end_time = inp.get("end_time")
    include_predictions = inp.get("include_predictions", True)

    if not video_url and not video_path:
        return {"error": "Either video_url or video_path is required"}

    is_warm_start = _requests_served > 0
    _requests_served += 1

    source = video_path if video_path else video_url
    cleanup_dir = None

    try:
        # Normalize time range.
        try:
            start_time, end_time = normalize_time_range(start_time, end_time)
        except ValueError as exc:
            return {"error": f"Invalid time range: {exc}"}

        # Prepare video (download if remote, trim if needed).
        prepared_path, transfer_metadata, cleanup_dir = prepare_video_input(
            source, start_time=start_time, end_time=end_time,
        )

        # Phase-level progress collector.
        phase_progress = []

        def progress_callback(update):
            phase_progress.append({
                "step": update.get("step"),
                "stage": update.get("stage"),
                "step_index": update.get("step_index"),
                "elapsed_seconds": update.get("elapsed_seconds"),
                "step_elapsed_seconds": update.get("step_elapsed_seconds"),
            })

        result = run_inference(str(prepared_path), progress_callback=progress_callback)

        # Extract timing.
        timing = result.get("timing", {})
        phase_timings = timing.get("phases", {})
        total_seconds = timing.get("total_seconds", 0.0)

        # GPU snapshot.
        gpu_snapshot = None
        try:
            snap = runtime_status_snapshot()
            gpu_snapshot = snap.get("gpu")
        except Exception:
            pass

        # Build Prometheus text.
        metrics_text = _format_prometheus_metrics(
            phase_timings=phase_timings,
            warmup_timings=_warmup_timings,
            warmup_total=_warmup_total_seconds,
            total_seconds=total_seconds,
            is_warm_start=is_warm_start,
            gpu_snapshot=gpu_snapshot,
        )

        # Enrich timing block.
        result["timing"]["warmup"] = {
            "total_seconds": _warmup_total_seconds,
            "phases": _warmup_timings,
        }
        result["timing"]["warm_start"] = is_warm_start
        result["timing"]["transfer"] = transfer_metadata
        result["metrics_text"] = metrics_text

        # Optionally strip the large predictions array for lighter responses.
        if not include_predictions:
            result.pop("predictions", None)
            result.pop("spectrum", None)

        return result

    except Exception as exc:
        logger.exception("Inference failed")
        return {"error": str(exc)}
    finally:
        if cleanup_dir is not None:
            import shutil
            shutil.rmtree(cleanup_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
