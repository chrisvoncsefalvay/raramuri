from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

import infer


def _make_model(*, has_text_feature: bool, model_name: str | None = "meta-llama/Llama-3.2-3B"):
    data = type("Data", (), {})()
    if has_text_feature:
        data.text_feature = type("TextFeature", (), {"model_name": model_name})()
    return type("Model", (), {"data": data})()


def test_text_embeddings_report_no_word_events_when_text_absent():
    model = _make_model(has_text_feature=False)
    events = pd.DataFrame([{"type": "Audio"}, {"type": "Video"}])

    status = infer.inspect_text_embedding_status(model, events, audio_only=False)

    assert status["status"] == "no_word_events"
    assert status["word_event_count"] == 0
    assert status["extractor_present"] is False


def test_text_embeddings_raise_when_words_exist_but_extractor_missing():
    model = _make_model(has_text_feature=False)
    events = pd.DataFrame([{"type": "Word", "text": "hello"}])

    with pytest.raises(RuntimeError, match="text embedding extractor/model is unavailable"):
        infer.inspect_text_embedding_status(model, events, audio_only=False)


def test_text_embeddings_report_available_when_words_and_extractor_exist():
    model = _make_model(has_text_feature=True)
    events = pd.DataFrame([{"type": "Word", "text": "hello"}])

    status = infer.inspect_text_embedding_status(model, events, audio_only=False)

    assert status["status"] == "available"
    assert status["word_event_count"] == 1
    assert status["extractor_present"] is True
    assert status["model_name"] == "meta-llama/Llama-3.2-3B"


def test_normalize_time_range_accepts_30_second_clip():
    start_time, end_time = infer.normalize_time_range("00:00:00", "00:00:30")

    assert start_time == "00:00:00"
    assert end_time == "00:00:30"


def test_normalize_time_range_rejects_inverted_range():
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        infer.normalize_time_range("30", "10")