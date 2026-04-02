"""Precision quality verification for Rarámuri optimisations.

Tests that BF16 and FP8 inference produce predictions within acceptable
tolerances of the FP32 reference, measured on final TRIBEv2 brain
activation maps (not intermediate hidden states).

These tests require a running container with GPU access and the test
fixture video available. Run via:

    docker exec -w /app container python -m pytest /repo/tests/test_precision.py -v

or from the host:

    docker exec -w /app -e TRIBE_CACHE=/tmp/test_cache container \
        python -m pytest /tests/test_precision.py -v
"""

import os
import pickle
import subprocess
import tempfile

import numpy as np
import pytest

FIXTURE_VIDEO = os.environ.get(
    "RARAMURI_TEST_VIDEO", "/repo/profiling/bluey-15s-90s.mp4"
)

# Thresholds derived from measured quality verification (see profiles/gb10/)
BF16_MAX_NRMSE = 0.001  # 0.1% — measured 0.022%
FP8_MAX_NRMSE = 0.01  # 1.0% — measured 0.429%
BF16_MIN_VERTEX_CORR_MEAN = 0.9999  # measured 0.99999
FP8_MIN_VERTEX_CORR_MEAN = 0.995  # measured 0.99800


def _run_inference_subprocess(dtype: str, quant: str, cache_dir: str) -> np.ndarray:
    """Run inference in a subprocess with a unique cache directory."""
    script = f"""
import sys; sys.path.insert(0, '/app')
import numpy as np, os, pickle
os.environ['RARAMURI_DISABLE_WHISPERX'] = '1'
os.environ['RARAMURI_NUM_WORKERS'] = '0'
os.environ['RARAMURI_PERSIST_EXTRACTOR_MODELS'] = '0'
os.environ['RARAMURI_VJEPA_DTYPE'] = '{dtype}'
os.environ['RARAMURI_VJEPA_QUANT'] = '{quant}'
os.environ['TRIBE_CACHE'] = '{cache_dir}'

from infer import load_model, patch_runtime_extractors
import infer as infer_mod
infer_mod._CACHED_MODEL = None
patch_runtime_extractors()
model, _ = load_model(reuse=False)
events = model.get_events_dataframe(video_path='{FIXTURE_VIDEO}')
preds, _ = model.predict(events=events, verbose=False)
with open('{cache_dir}/preds.pkl', 'wb') as f:
    pickle.dump(preds.astype(np.float64), f)
"""
    result = subprocess.run(
        ["python", "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd="/app",
    )
    if result.returncode != 0:
        pytest.fail(f"Inference failed (dtype={dtype}, quant={quant}):\n{result.stderr[-1000:]}")

    with open(os.path.join(cache_dir, "preds.pkl"), "rb") as f:
        return pickle.load(f)


def _compare_predictions(ref: np.ndarray, test: np.ndarray) -> dict:
    """Compute quality metrics between two prediction arrays."""
    diff = test - ref
    rmse = np.sqrt(np.mean(diff**2))
    signal_range = np.ptp(ref)
    nrmse = rmse / signal_range if signal_range > 0 else 0.0

    n_verts = ref.shape[1]
    vertex_corr = np.array(
        [np.corrcoef(ref[:, i], test[:, i])[0, 1] for i in range(n_verts)]
    )
    vertex_corr = np.nan_to_num(vertex_corr, nan=1.0)

    return {
        "nrmse": nrmse,
        "vertex_corr_mean": vertex_corr.mean(),
        "vertex_corr_p5": np.percentile(vertex_corr, 5),
    }


@pytest.fixture(scope="module")
def fp32_predictions():
    """FP32 reference predictions (computed once per test module)."""
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip(f"Test fixture not found: {FIXTURE_VIDEO}")
    with tempfile.TemporaryDirectory(prefix="raramuri_test_fp32_") as d:
        return _run_inference_subprocess("fp32", "none", d)


class TestBF16Quality:
    """BF16 should be indistinguishable from FP32 on final predictions."""

    @pytest.fixture(scope="class")
    def bf16_predictions(self):
        with tempfile.TemporaryDirectory(prefix="raramuri_test_bf16_") as d:
            return _run_inference_subprocess("bf16", "none", d)

    def test_nrmse_below_threshold(self, fp32_predictions, bf16_predictions):
        metrics = _compare_predictions(fp32_predictions, bf16_predictions)
        assert metrics["nrmse"] < BF16_MAX_NRMSE, (
            f"BF16 NRMSE {metrics['nrmse']:.6f} exceeds threshold {BF16_MAX_NRMSE}"
        )

    def test_vertex_correlation(self, fp32_predictions, bf16_predictions):
        metrics = _compare_predictions(fp32_predictions, bf16_predictions)
        assert metrics["vertex_corr_mean"] > BF16_MIN_VERTEX_CORR_MEAN, (
            f"BF16 vertex correlation {metrics['vertex_corr_mean']:.8f} "
            f"below threshold {BF16_MIN_VERTEX_CORR_MEAN}"
        )

    def test_prediction_shape_unchanged(self, fp32_predictions, bf16_predictions):
        assert fp32_predictions.shape == bf16_predictions.shape


class TestFP8Quality:
    """FP8 should be within acceptable tolerance for preview/iteration use."""

    @pytest.fixture(scope="class")
    def fp8_predictions(self):
        with tempfile.TemporaryDirectory(prefix="raramuri_test_fp8_") as d:
            return _run_inference_subprocess("bf16", "fp8", d)

    def test_nrmse_below_threshold(self, fp32_predictions, fp8_predictions):
        metrics = _compare_predictions(fp32_predictions, fp8_predictions)
        assert metrics["nrmse"] < FP8_MAX_NRMSE, (
            f"FP8 NRMSE {metrics['nrmse']:.6f} exceeds threshold {FP8_MAX_NRMSE}"
        )

    def test_vertex_correlation(self, fp32_predictions, fp8_predictions):
        metrics = _compare_predictions(fp32_predictions, fp8_predictions)
        assert metrics["vertex_corr_mean"] > FP8_MIN_VERTEX_CORR_MEAN, (
            f"FP8 vertex correlation {metrics['vertex_corr_mean']:.8f} "
            f"below threshold {FP8_MIN_VERTEX_CORR_MEAN}"
        )

    def test_prediction_shape_unchanged(self, fp32_predictions, fp8_predictions):
        assert fp32_predictions.shape == fp8_predictions.shape
