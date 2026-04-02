"""Composite video renderer: original video + glass brain activations + captions + spectrum.

Given an inference result JSON and the source video, renders a side-by-side MP4:
  Left:  original video with timestamp overlay
  Right: nilearn glass brain of interpolated cortical activations,
         progressively drawn audio spectrum, and caption text

The 20,484-vertex predictions (fsaverage5 surface, 1 Hz) are linearly interpolated
to the video frame rate, projected to MNI volume, and rendered as glass brain frames
via nilearn.plotting.plot_glass_brain.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting
from nilearn.surface import load_surf_mesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surface → Volume projection (cached once per process)
# ---------------------------------------------------------------------------

_PROJECTION_CACHE: dict = {}


def _get_projection_grid():
    """Build the surface-to-volume projection mapping for fsaverage5."""
    if "voxel_map" in _PROJECTION_CACHE:
        return _PROJECTION_CACHE["voxel_map"], _PROJECTION_CACHE["shape"], _PROJECTION_CACHE["affine"]

    fs5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lh_coords, _ = load_surf_mesh(fs5["pial_left"])
    rh_coords, _ = load_surf_mesh(fs5["pial_right"])
    all_coords = np.vstack([lh_coords, rh_coords])  # (20484, 3)

    resolution = 3  # mm
    affine = np.diag([resolution, resolution, resolution, 1.0])
    affine[:3, 3] = [-90, -126, -72]
    shape = (61, 73, 61)

    voxel_ijk = np.round((all_coords - affine[:3, 3]) / resolution).astype(int)
    # Build a dict: (i,j,k) -> list of vertex indices
    voxel_map: dict[tuple[int, int, int], list[int]] = {}
    for vidx, (vi, vj, vk) in enumerate(voxel_ijk):
        if 0 <= vi < shape[0] and 0 <= vj < shape[1] and 0 <= vk < shape[2]:
            key = (int(vi), int(vj), int(vk))
            voxel_map.setdefault(key, []).append(vidx)

    _PROJECTION_CACHE["voxel_map"] = voxel_map
    _PROJECTION_CACHE["shape"] = shape
    _PROJECTION_CACHE["affine"] = affine
    logger.info("Projection grid cached: %d voxels from 20484 vertices", len(voxel_map))
    return voxel_map, shape, affine


def surface_to_volume(vertex_data: np.ndarray) -> nib.Nifti1Image:
    """Project a 20484-element surface vector to a 3D NIfTI volume."""
    voxel_map, shape, affine = _get_projection_grid()
    vol = np.zeros(shape, dtype=np.float32)
    for (vi, vj, vk), indices in voxel_map.items():
        vol[vi, vj, vk] = np.mean(vertex_data[indices])
    return nib.Nifti1Image(vol, affine)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_composite_frame(
    *,
    video_frame: np.ndarray,
    activation_map: np.ndarray,
    spectrum_history: np.ndarray,
    timestamp: float,
    duration: float,
    caption: str | None,
    vmin: float,
    vmax: float,
    output_path: str,
    dpi: int = 100,
    width_px: int = 1920,
    height_px: int = 540,
) -> None:
    """Render one composite frame to disk."""
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[4, 1],
                           wspace=0.02, hspace=0.08,
                           left=0.01, right=0.99, top=0.95, bottom=0.03)

    # --- Left: original video frame with timestamp ---
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.imshow(video_frame)
    ax_video.set_axis_off()
    minutes = int(timestamp) // 60
    seconds = timestamp - minutes * 60
    ts_text = f"{minutes:02d}:{seconds:05.2f}"
    ax_video.text(
        0.02, 0.96, ts_text,
        transform=ax_video.transAxes,
        fontsize=14, fontweight="bold",
        color="white", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        verticalalignment="top",
    )

    # --- Right: glass brain ---
    ax_brain = fig.add_subplot(gs[0, 1])
    vol_img = surface_to_volume(activation_map)
    # Use threshold at 10th percentile of abs values to reduce noise
    abs_vals = np.abs(activation_map)
    threshold = np.percentile(abs_vals[abs_vals > 0], 10) if np.any(abs_vals > 0) else 0
    display = plotting.plot_glass_brain(
        vol_img,
        display_mode="lyrz",
        colorbar=False,
        plot_abs=False,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        axes=ax_brain,
        black_bg=True,
    )
    ax_brain.set_axis_off()

    # --- Bottom left: captions ---
    ax_caption = fig.add_subplot(gs[1, 0])
    ax_caption.set_facecolor("black")
    ax_caption.set_axis_off()
    if caption:
        ax_caption.text(
            0.5, 0.5, caption,
            transform=ax_caption.transAxes,
            fontsize=11, color="white", family="sans-serif",
            horizontalalignment="center", verticalalignment="center",
            wrap=True,
        )

    # --- Bottom right: spectrum ---
    ax_spec = fig.add_subplot(gs[1, 1])
    ax_spec.set_facecolor("black")
    if spectrum_history is not None and spectrum_history.shape[0] > 0:
        ax_spec.imshow(
            spectrum_history.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="bilinear",
        )
        # Draw progress line
        progress_x = spectrum_history.shape[0] - 1
        ax_spec.axvline(x=progress_x, color="white", linewidth=0.5, alpha=0.6)
    ax_spec.set_axis_off()

    fig.savefig(output_path, dpi=dpi, facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video extraction helpers
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: str, fps: float, output_dir: str) -> int:
    """Extract frames from video at the given fps using ffmpeg. Returns frame count."""
    pattern = os.path.join(output_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[:500]}")
    frames = sorted(Path(output_dir).glob("frame_*.png"))
    return len(frames)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    return float(result.stdout.strip())


def assemble_video(frame_dir: str, fps: float, output_path: str) -> None:
    """Assemble PNG frames into an MP4."""
    pattern = os.path.join(frame_dir, "composite_%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Extract the actual error lines (skip config noise)
        err_lines = [l for l in result.stderr.splitlines() if not l.startswith("  ") and "configuration:" not in l]
        raise RuntimeError(f"ffmpeg assembly failed (exit {result.returncode}): " + "\n".join(err_lines[-10:]))


def add_audio_track(video_path: str, source_video: str, output_path: str) -> None:
    """Mux the audio from source_video into the rendered video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", source_video,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # No audio track is not fatal
        logger.warning("Audio mux failed (source may lack audio): %s", result.stderr[:200])
        shutil.copy2(video_path, output_path)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_predictions(predictions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Linearly interpolate predictions from source_fps (typically 1 Hz) to target_fps.

    Args:
        predictions: (n_timepoints, n_vertices) array at source_fps
        source_fps: original frame rate (e.g. 1.0)
        target_fps: desired output frame rate (e.g. 24.0)

    Returns:
        Interpolated array at target_fps.
    """
    n_src = predictions.shape[0]
    duration = n_src / source_fps
    n_dst = int(duration * target_fps)

    src_times = np.arange(n_src) / source_fps
    dst_times = np.arange(n_dst) / target_fps

    # Vectorized linear interpolation across all vertices
    interpolated = np.zeros((n_dst, predictions.shape[1]), dtype=np.float32)
    for i, t in enumerate(dst_times):
        # Find bounding source frames
        idx = t * source_fps
        lo = int(np.floor(idx))
        hi = min(lo + 1, n_src - 1)
        alpha = idx - lo
        interpolated[i] = (1 - alpha) * predictions[lo] + alpha * predictions[hi]

    return interpolated


def interpolate_spectrum(spectrum: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Interpolate spectrum the same way as predictions."""
    return interpolate_predictions(spectrum, source_fps, target_fps)


# ---------------------------------------------------------------------------
# Caption lookup
# ---------------------------------------------------------------------------

def caption_at_time(captions: list[dict], timestamp: float) -> str | None:
    """Find the caption active at the given timestamp."""
    for cap in captions:
        start = cap.get("start", 0)
        end = start + cap.get("duration", 0)
        if start <= timestamp < end:
            return cap.get("text", "")
    return None


# ---------------------------------------------------------------------------
# Main render pipeline
# ---------------------------------------------------------------------------

def render_composite_video(
    *,
    result: dict,
    video_path: str,
    output_path: str,
    fps: float = 24.0,
    dpi: int = 100,
    width_px: int = 1920,
    height_px: int = 540,
    progress_callback=None,
) -> dict:
    """Render a composite visualization video from inference results.

    Args:
        result: Inference result dict with predictions, spectrum, segments, captions.
        video_path: Path to the source video file.
        output_path: Where to write the final MP4.
        fps: Output video frame rate (predictions are interpolated to this).
        dpi: Matplotlib DPI for rendered frames.
        width_px: Output width in pixels.
        height_px: Output height in pixels.
        progress_callback: Optional callable(dict) for progress updates.

    Returns:
        Dict with render metadata (duration, frame_count, output_path, timing).
    """
    t_start = time.monotonic()
    timings = {}

    predictions = np.array(result["predictions"], dtype=np.float32)
    spectrum = np.array(result.get("spectrum", []), dtype=np.float32) if result.get("spectrum") else None
    captions = result.get("captions", [])
    segments = result.get("segments", [])

    n_timepoints = predictions.shape[0]
    source_fps = 1.0  # predictions are at 1 Hz
    duration = n_timepoints / source_fps

    # Global color scale from predictions
    vmin = float(np.percentile(predictions, 2))
    vmax = float(np.percentile(predictions, 98))

    # Interpolate to target fps
    t0 = time.monotonic()
    interp_preds = interpolate_predictions(predictions, source_fps, fps)
    interp_spec = interpolate_spectrum(spectrum, source_fps, fps) if spectrum is not None else None
    n_frames = interp_preds.shape[0]
    timings["interpolation"] = round(time.monotonic() - t0, 3)
    logger.info("Interpolated %d -> %d frames (%.1f fps)", n_timepoints, n_frames, fps)

    workdir = tempfile.mkdtemp(prefix="raramuri-viz-")
    try:
        # Extract video frames
        t0 = time.monotonic()
        src_frame_dir = os.path.join(workdir, "src_frames")
        os.makedirs(src_frame_dir)
        extract_video_frames(video_path, fps, src_frame_dir)
        src_frames = sorted(Path(src_frame_dir).glob("frame_*.png"))
        timings["frame_extraction"] = round(time.monotonic() - t0, 3)
        logger.info("Extracted %d source frames", len(src_frames))

        # Pre-cache projection grid
        t0 = time.monotonic()
        _get_projection_grid()
        timings["projection_grid"] = round(time.monotonic() - t0, 3)

        # Render composite frames
        t0 = time.monotonic()
        comp_frame_dir = os.path.join(workdir, "comp_frames")
        os.makedirs(comp_frame_dir)

        for i in range(n_frames):
            timestamp = i / fps

            # Load source video frame (clamp to available frames)
            src_idx = min(i, len(src_frames) - 1)
            video_frame = plt.imread(str(src_frames[src_idx]))

            # Spectrum history up to current frame
            spec_history = interp_spec[:i + 1] if interp_spec is not None else None

            # Current caption
            caption = caption_at_time(captions, timestamp)

            comp_path = os.path.join(comp_frame_dir, f"composite_{i + 1:06d}.png")
            render_composite_frame(
                video_frame=video_frame,
                activation_map=interp_preds[i],
                spectrum_history=spec_history,
                timestamp=timestamp,
                duration=duration,
                caption=caption,
                vmin=vmin,
                vmax=vmax,
                output_path=comp_path,
                dpi=dpi,
                width_px=width_px,
                height_px=height_px,
            )

            if progress_callback and i % max(1, n_frames // 20) == 0:
                progress_callback({
                    "step": "rendering",
                    "progress_ratio": i / n_frames,
                    "frame": i,
                    "total_frames": n_frames,
                })

        timings["frame_rendering"] = round(time.monotonic() - t0, 3)
        logger.info("Rendered %d composite frames in %.1fs", n_frames, timings["frame_rendering"])

        # Assemble video
        t0 = time.monotonic()
        raw_video = os.path.join(workdir, "raw.mp4")
        assemble_video(comp_frame_dir, fps, raw_video)
        timings["assembly"] = round(time.monotonic() - t0, 3)

        # Add audio from original
        t0 = time.monotonic()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        add_audio_track(raw_video, video_path, output_path)
        timings["audio_mux"] = round(time.monotonic() - t0, 3)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    total_seconds = round(time.monotonic() - t_start, 3)
    timings["total"] = total_seconds
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    logger.info(
        "Composite video rendered: %d frames, %.1fs duration, %s, %.1fs render time",
        n_frames, duration, output_path, total_seconds,
    )

    return {
        "output_path": output_path,
        "duration_seconds": duration,
        "frame_count": n_frames,
        "fps": fps,
        "resolution": f"{width_px}x{height_px}",
        "file_size_bytes": file_size,
        "timing": timings,
    }
