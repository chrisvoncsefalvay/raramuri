"""End-to-end inference smoke tests.

Verifies that the inference pipeline produces valid output with the
expected shape and structure. Does not check prediction quality
(see test_precision.py for that).

Requires a running container with GPU access.
"""

import json
import os
import subprocess
import tempfile

import pytest

FIXTURE_VIDEO = os.environ.get(
    "RARAMURI_TEST_VIDEO", "/repo/profiling/bluey-15s-90s.mp4"
)


@pytest.fixture(scope="module")
def skip_if_no_fixture():
    if not os.path.exists(FIXTURE_VIDEO):
        pytest.skip(f"Test fixture not found: {FIXTURE_VIDEO}")


def _run_infer(env_overrides: dict | None = None) -> dict:
    """Run infer.py and return the parsed JSON result."""
    env = {
        "RARAMURI_DISABLE_WHISPERX": "1",
        "RARAMURI_NUM_WORKERS": "0",
        "RARAMURI_SKIP_METRICS": "1",
        "RARAMURI_VJEPA_DTYPE": "bf16",
    }
    if env_overrides:
        env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        cmd = ["python", "/app/infer.py", FIXTURE_VIDEO, "-o", output_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, **env},
            cwd="/app",
        )
        if result.returncode != 0:
            pytest.fail(f"infer.py failed:\n{result.stderr[-1000:]}")

        with open(output_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


class TestInferenceOutput:
    """Verify the structure and shape of inference output."""

    @pytest.fixture(scope="class")
    def result(self, skip_if_no_fixture):
        return _run_infer()

    def test_has_predictions(self, result):
        assert "predictions" in result
        assert len(result["predictions"]) > 0

    def test_prediction_shape(self, result):
        shape = result["shape"]
        assert len(shape) == 2
        assert shape[0] > 0  # timesteps
        assert shape[1] == 20484  # vertices

    def test_has_timing(self, result):
        timing = result["timing"]
        assert "total_seconds" in timing
        assert "phases" in timing
        assert timing["total_seconds"] > 0

    def test_has_segments(self, result):
        assert "segments" in result
        for seg in result["segments"]:
            assert "start" in seg
            assert "duration" in seg

    def test_has_event_types(self, result):
        events = result["event_types"]
        assert "Audio" in events
        assert "Video" in events


class TestPrecisionModes:
    """Verify that different precision modes run without errors."""

    def test_fp32_runs(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_VJEPA_DTYPE": "fp32"})
        assert result["shape"][1] == 20484

    def test_bf16_runs(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_VJEPA_DTYPE": "bf16"})
        assert result["shape"][1] == 20484

    def test_fp16_runs(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_VJEPA_DTYPE": "fp16"})
        assert result["shape"][1] == 20484


class TestParallelModes:
    """Verify that parallel extractor modes run without errors."""

    def test_mode_0_sequential(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_PARALLEL_EXTRACTORS": "0"})
        assert result["shape"][1] == 20484

    def test_mode_1_light_parallel(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_PARALLEL_EXTRACTORS": "1"})
        assert result["shape"][1] == 20484

    def test_mode_2_aggressive_parallel(self, skip_if_no_fixture):
        result = _run_infer({"RARAMURI_PARALLEL_EXTRACTORS": "2"})
        assert result["shape"][1] == 20484
