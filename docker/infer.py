"""Cloud inference script for TRIBE v2.

Usage:
    python infer.py <youtube_url> [--output results.json]

Downloads video, runs full TRIBE v2 inference (audio + video + text),
computes mel spectrogram and extracts captions,
writes a JSON bundle with predictions, segments, metrics, spectrum, and captions.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

# Guard stderr for tqdm
if sys.stderr is None or sys.stderr.closed:
    sys.stderr = io.StringIO()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_ENV_DEFAULTS = {
    "HF_HOME": "/models/hf",
    "TRIBE_CACHE": "/models/tribe",
}

HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")

_CACHED_MODEL = None


def _set_extractor_model_persistence(enabled: bool) -> None:
    """Enable/disable extractor model persistence by patching tribev2 free hook."""
    import tribev2.main as tribev2_main

    if not hasattr(tribev2_main, '_raramuri_original_free_extractor_model'):
        tribev2_main._raramuri_original_free_extractor_model = tribev2_main._free_extractor_model

    if enabled:
        def _keep_extractor_model(extractor):
            logger.info('Persisting extractor model for %s', extractor.__class__.__name__)
            return None

        tribev2_main._free_extractor_model = _keep_extractor_model
        tribev2_main._raramuri_persist_models = True
        logger.info('Extractor model persistence enabled')
    else:
        original = getattr(tribev2_main, '_raramuri_original_free_extractor_model', None)
        if original is not None:
            tribev2_main._free_extractor_model = original
        tribev2_main._raramuri_persist_models = False
        logger.info('Extractor model persistence disabled')


def ensure_hf_token_env() -> str:
    """Normalize Hugging Face token env vars and return the resolved token.

    Some libraries read only one of the supported env var names; mirror the
    first non-empty value into all aliases to make auth deterministic.
    """
    token = ""
    token_source = None
    for name in HF_TOKEN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            token = value
            token_source = name
            break

    if not token:
        return ""

    for name in HF_TOKEN_ENV_NAMES:
        os.environ[name] = token

    logger.info("HF auth token detected via %s and mirrored to alias env vars", token_source)
    return token


def _normalize_runtime_cache_path(key: str, default: str) -> Path:
    """Resolve runtime cache roots, handling broken absolute symlinks in bind mounts.

    Host-mounted directories sometimes contain absolute symlinks that are valid on
    the host but broken inside the container, such as /models/hf ->
    /home/user/models/.cache/huggingface. For HF_HOME, redirect to an in-mount
    fallback under /models/.cache/huggingface when that happens.
    """
    path = Path(os.environ.setdefault(key, default))

    if key == 'HF_HOME' and path.is_symlink() and not path.exists():
        fallback = path.parent / '.cache' / 'huggingface'
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(fallback)
        logger.warning('Broken HF_HOME symlink inside container: %s -> using %s', path, fallback)
        return fallback

    if os.path.lexists(path):
        if not path.is_dir() and not path.is_symlink():
            raise RuntimeError(f"Runtime path exists but is not a directory: {path}")
        return path

    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_proc_status_kb(field: str) -> int | None:
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(field + ':'):
                parts = line.split()
                return int(parts[1])
    except Exception:
        return None
    return None


def _gpu_memory_snapshot() -> dict | None:
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        return {
            'allocated_mb': round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            'reserved_mb': round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
            'max_allocated_mb': round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
            'max_reserved_mb': round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2),
        }
    except Exception:
        return None


def runtime_status_snapshot() -> dict:
    """Return a lightweight runtime snapshot for observability endpoints."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    vmrss_kb = _read_proc_status_kb('VmRSS')
    vmsize_kb = _read_proc_status_kb('VmSize')
    return {
        'process': {
            'rss_max_mb': round(rss_kb / 1024, 2),
            'rss_current_mb': None if vmrss_kb is None else round(vmrss_kb / 1024, 2),
            'vmsize_mb': None if vmsize_kb is None else round(vmsize_kb / 1024, 2),
        },
        'gpu': _gpu_memory_snapshot(),
    }


def log_memory_state(label: str) -> None:
    snapshot = runtime_status_snapshot()
    logger.info(
        'MEMORY %s rss_max_mb=%.2f rss_current_mb=%s vmsize_mb=%s gpu=%s',
        label,
        snapshot['process']['rss_max_mb'],
        snapshot['process']['rss_current_mb'],
        snapshot['process']['vmsize_mb'],
        snapshot['gpu'],
    )


@contextmanager
def profiled_phase(name: str):
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
    except Exception:
        pass
    log_memory_state(f'{name}:start')
    started = time.monotonic()
    try:
        yield
    finally:
        elapsed = round(time.monotonic() - started, 3)
        log_memory_state(f'{name}:end')
        logger.info('PHASE %s seconds=%.3f', name, elapsed)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def _parakeet_nemo_output_to_dataframe(sample) -> "pd.DataFrame":
    """Convert a NeMo ASR transcription output to the transcript-compatible dataframe contract.

    Expected output columns: text, start, duration, sequence_id, sentence
    """
    import pandas as pd

    word_rows = []
    segment_rows = sample.timestamp.get('segment', []) if hasattr(sample, 'timestamp') else []
    words = sample.timestamp.get('word', []) if hasattr(sample, 'timestamp') else []
    for word in words:
        text = str(word.get('word', '')).strip()
        if not text:
            continue
        start = float(word.get('start', 0.0))
        end = float(word.get('end', start))
        sequence_id = 0
        sentence = text
        for seg_idx, seg in enumerate(segment_rows):
            seg_start = float(seg.get('start', 0.0))
            seg_end = float(seg.get('end', seg_start))
            if start >= seg_start and end <= seg_end + 1e-6:
                sequence_id = seg_idx
                sentence = str(seg.get('segment', text)).strip()
                break
        word_rows.append({
            'text': text,
            'start': start,
            'duration': max(0.0, end - start),
            'sequence_id': sequence_id,
            'sentence': sentence,
        })
    return pd.DataFrame(word_rows, columns=['text', 'start', 'duration', 'sequence_id', 'sentence'])


# ---------------------------------------------------------------------------
# Early/parallel Parakeet transcription
# ---------------------------------------------------------------------------
# When RARAMURI_TRANSCRIPT_BACKEND=parakeet, we launch transcription on CPU
# in a background thread *before* the GPU model load, so it runs fully
# parallel with TRIBE model loading and event construction.  The monkeypatched
# _get_transcript_from_audio just waits on the precomputed future.
# ---------------------------------------------------------------------------

_PARAKEET_FUTURES: dict[str, "concurrent.futures.Future"] = {}
_PARAKEET_EXECUTOR: "concurrent.futures.ThreadPoolExecutor | None" = None
_PARAKEET_MODEL = None  # cached NeMo ASR model for Parakeet persistence


def _get_parakeet_model(model_name: str):
    """Return cached Parakeet model, loading on first call."""
    global _PARAKEET_MODEL
    if _PARAKEET_MODEL is not None:
        return _PARAKEET_MODEL
    import nemo.collections.asr as nemo_asr
    t0 = time.monotonic()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location='cpu')
    logger.info('Parakeet model loaded: %.3fs model=%s', time.monotonic() - t0, model_name)
    _PARAKEET_MODEL = model
    return model


def _prepare_parakeet_wav(wav_path: str) -> str:
    """Normalize arbitrary input WAVs into a mono 16 kHz PCM file for Parakeet."""
    wav = Path(wav_path).expanduser()
    stat = wav.stat()
    cache_dir = Path(os.environ.get("RARAMURI_PARAKEET_AUDIO_CACHE_DIR", "/tmp/raramuri-parakeet"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = f"{wav.resolve(strict=False)}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    prepared_wav = cache_dir / f"{wav.stem}-{digest}-mono16k.wav"
    if prepared_wav.exists():
        return str(prepared_wav)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(wav),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(prepared_wav),
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Parakeet audio normalized: %s -> %s", wav, prepared_wav)
    return str(prepared_wav)


def _parakeet_audio_cache_key(wav_path: str) -> tuple[str, str]:
    """Return a stable cache key for normalized Parakeet audio plus the normalized path."""
    prepared_wav = _prepare_parakeet_wav(wav_path)
    digest = hashlib.sha256()
    with open(prepared_wav, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), prepared_wav


def _run_parakeet_cpu(wav_path: str, model_name: str) -> "pd.DataFrame":
    """Run Parakeet transcription on CPU.  Designed to execute in a thread."""
    import torch
    logger.info('Parakeet background: starting model=%s wav=%s device=cpu', model_name, wav_path)
    t0 = time.monotonic()
    model = _get_parakeet_model(model_name)
    load_elapsed = time.monotonic() - t0

    prepared_wav = _prepare_parakeet_wav(wav_path)
    t1 = time.monotonic()
    out = model.transcribe([prepared_wav], timestamps=True)[0]
    transcribe_elapsed = time.monotonic() - t1

    df = _parakeet_nemo_output_to_dataframe(out)
    transcript_text = " ".join(str(text).strip() for text in df.get("text", []) if str(text).strip())
    total = time.monotonic() - t0
    logger.info(
        'Parakeet background: done words=%d load=%.3fs transcribe=%.3fs total=%.3fs',
        len(df), load_elapsed, transcribe_elapsed, total,
    )
    logger.info("Parakeet transcript text: %s", transcript_text or "<empty>")
    return df


def start_early_parakeet(video_path: str) -> None:
    """Kick off Parakeet transcription in a background thread.

    Call this as early as possible in run_inference() — before GPU model load.
    The WAV is extracted synchronously (fast, ~0.1s via ffmpeg) and the actual
    NeMo model load + transcribe runs on CPU in a thread, fully parallel with
    GPU work.
    """
    import concurrent.futures

    global _PARAKEET_EXECUTOR

    transcript_backend = os.environ.get('RARAMURI_TRANSCRIPT_BACKEND', '').strip().lower()
    if transcript_backend != 'parakeet':
        return

    wav_path = extract_audio_wav(video_path)
    model_name = os.environ.get('RARAMURI_PARAKEET_MODEL', 'nvidia/parakeet-tdt-0.6b-v2')
    cache_key, prepared_wav = _parakeet_audio_cache_key(wav_path)

    if _PARAKEET_EXECUTOR is None:
        _PARAKEET_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='parakeet')

    future = _PARAKEET_EXECUTOR.submit(_run_parakeet_cpu, prepared_wav, model_name)
    _PARAKEET_FUTURES[cache_key] = future
    logger.info('Parakeet early transcription submitted: wav=%s cache_key=%s', prepared_wav, cache_key[:12])


def _resolve_torch_dtype(name: str):
    """Map a dtype string to a torch dtype, or None for 'fp32'/'default'."""
    import torch
    _MAP = {
        'fp32': None, 'float32': None, 'default': None, '': None,
        'fp16': torch.float16, 'float16': torch.float16, 'half': torch.float16,
        'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
    }
    return _MAP.get(name.strip().lower())


def _apply_fp8_quantization(model, label: str) -> None:
    """Apply FP8 dynamic activation + FP8 weight quantization via torchao."""
    try:
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        logger.info('Applied FP8 dynamic quantization to %s', label)
    except Exception as exc:
        logger.warning('FP8 quantization failed for %s: %s', label, exc)


def _patch_extractor_precision() -> dict:
    """Patch neuralset extractor model loading to use configured precision.

    Env vars (all optional, default=bf16):
        RARAMURI_VJEPA_DTYPE   — video model dtype: fp32, fp16, bf16 (default: bf16)
        RARAMURI_AUDIO_DTYPE   — audio model dtype: fp32, fp16, bf16 (default: bf16)
        RARAMURI_TEXT_DTYPE    — text model dtype: fp32, fp16, bf16  (default: bf16)
        RARAMURI_VJEPA_QUANT   — video model quantization: fp8, none (default: none)
        RARAMURI_AUDIO_QUANT   — audio model quantization: fp8, none (default: none)
        RARAMURI_TEXT_QUANT    — text model quantization: fp8, none  (default: none)

    Returns dict of applied settings for logging.
    """
    import torch

    vjepa_dtype_name = os.environ.get('RARAMURI_VJEPA_DTYPE', 'bf16').strip().lower()
    audio_dtype_name = os.environ.get('RARAMURI_AUDIO_DTYPE', 'bf16').strip().lower()
    text_dtype_name = os.environ.get('RARAMURI_TEXT_DTYPE', 'bf16').strip().lower()

    vjepa_quant = os.environ.get('RARAMURI_VJEPA_QUANT', 'none').strip().lower()
    audio_quant = os.environ.get('RARAMURI_AUDIO_QUANT', 'none').strip().lower()
    text_quant = os.environ.get('RARAMURI_TEXT_QUANT', 'none').strip().lower()

    vjepa_dtype = _resolve_torch_dtype(vjepa_dtype_name)
    audio_dtype = _resolve_torch_dtype(audio_dtype_name)
    text_dtype = _resolve_torch_dtype(text_dtype_name)

    settings = {
        'vjepa_dtype': vjepa_dtype_name or 'fp32',
        'vjepa_quant': vjepa_quant,
        'audio_dtype': audio_dtype_name or 'fp32',
        'audio_quant': audio_quant,
        'text_dtype': text_dtype_name or 'fp32',
        'text_quant': text_quant,
    }

    any_patch = (vjepa_dtype is not None or vjepa_quant == 'fp8'
                 or audio_dtype is not None or audio_quant == 'fp8'
                 or text_dtype is not None or text_quant == 'fp8')
    if not any_patch:
        logger.info('Extractor precision: all fp32, no quantization')
        return settings

    # Global lock for serializing HF model loading across threads.
    # HF Transformers' safetensors lazy loading uses meta tensor placeholders
    # that race when two threads call from_pretrained concurrently, causing
    # "Cannot copy out of meta tensor" errors.  Forwards are thread-safe.
    import threading as _threading
    _hf_load_lock = _threading.Lock()

    # --- Patch VJEPA2 video model ---
    try:
        from neuralset.extractors.video import _HFVideoModel
        if not getattr(_HFVideoModel, '_raramuri_precision_patched', False):
            _original_video_init = _HFVideoModel.__init__

            def _precision_video_init(self, *args, **kwargs):
                with _hf_load_lock:
                    _original_video_init(self, *args, **kwargs)
                    if vjepa_dtype is not None:
                        self.model = self.model.to(dtype=vjepa_dtype)
                        logger.info('VJEPA2 model cast to %s', vjepa_dtype)
                    if vjepa_quant == 'fp8':
                        _apply_fp8_quantization(self.model, 'VJEPA2')

            _HFVideoModel.__init__ = _precision_video_init
            _HFVideoModel._raramuri_precision_patched = True
            logger.info('Patched VJEPA2 precision: dtype=%s quant=%s', vjepa_dtype_name, vjepa_quant)
    except Exception as exc:
        logger.warning('VJEPA2 precision patch failed: %s', exc)

    # --- Patch Wav2VecBert audio model ---
    if audio_dtype is not None or audio_quant == 'fp8':
        try:
            import neuralset.extractors.audio as audio_module

            if not getattr(audio_module.Wav2VecBert, '_raramuri_precision_patched', False):
                _original_process_wav = audio_module.Wav2VecBert._process_wav
                _original_audio_model_prop = audio_module.Wav2VecBert.model.fget

                @property
                def _locked_audio_model(self):
                    """Wrap audio model property with load lock + optional FP8."""
                    with _hf_load_lock:
                        model = _original_audio_model_prop(self)
                    if audio_quant == 'fp8' and not getattr(model, '_raramuri_fp8_applied', False):
                        with _hf_load_lock:
                            _apply_fp8_quantization(model, 'Wav2VecBert')
                            model._raramuri_fp8_applied = True
                    return model

                audio_module.Wav2VecBert.model = _locked_audio_model

                def _precision_process_wav(self, wav):
                    """Wrap _process_wav with autocast for mixed-precision audio inference."""
                    import torch
                    if audio_dtype is not None and torch.cuda.is_available():
                        with torch.autocast('cuda', dtype=audio_dtype):
                            return _original_process_wav(self, wav)
                    return _original_process_wav(self, wav)

                audio_module.Wav2VecBert._process_wav = _precision_process_wav
                audio_module.Wav2VecBert._raramuri_precision_patched = True
                logger.info('Patched Wav2VecBert precision: dtype=%s quant=%s (load-locked)', audio_dtype_name, audio_quant)
        except Exception as exc:
            logger.warning('Wav2VecBert precision patch failed: %s', exc)

    # --- Patch HuggingFaceText (Llama) model ---
    if text_dtype is not None or text_quant == 'fp8':
        try:
            import neuralset.extractors.text as text_module

            if not getattr(text_module.HuggingFaceText, '_raramuri_precision_patched', False):
                _original_text_get_data = text_module.HuggingFaceText._get_data

                # For text, we wrap _get_data with autocast since the model is loaded
                # inside _get_data via self.model property and the forward happens there.
                # But the model is already patched via dtype= on from_pretrained for VJEPA2.
                # For text, the model property loads it on demand — we wrap the forward.
                _original_text_model_prop = text_module.HuggingFaceText.model.fget

                @property
                def _precision_text_model(self):
                    with _hf_load_lock:
                        model = _original_text_model_prop(self)
                        if not getattr(model, '_raramuri_dtype_applied', False):
                            if text_dtype is not None:
                                model = model.to(dtype=text_dtype)
                                logger.info('HuggingFaceText model cast to %s', text_dtype)
                            if text_quant == 'fp8':
                                _apply_fp8_quantization(model, 'HuggingFaceText')
                            model._raramuri_dtype_applied = True
                    return model

                text_module.HuggingFaceText.model = _precision_text_model
                text_module.HuggingFaceText._raramuri_precision_patched = True
                logger.info('Patched HuggingFaceText precision: dtype=%s quant=%s', text_dtype_name, text_quant)
        except Exception as exc:
            logger.warning('HuggingFaceText precision patch failed: %s', exc)

    logger.info('Extractor precision settings: %s', settings)
    return settings


def patch_runtime_extractors() -> None:
    """Apply runtime patches/tuning for the current container process only."""
    detailed_profile = os.environ.get('RARAMURI_DETAILED_PROFILE', '0') == '1'
    # Backwards compatibility: treat the legacy disable flag as explicit disabled backend.
    transcript_backend = os.environ.get('RARAMURI_TRANSCRIPT_BACKEND', '').strip().lower()
    if os.environ.get('RARAMURI_DISABLE_WHISPERX', '0') == '1' and not transcript_backend:
        transcript_backend = 'disabled'

    # Apply precision/quantization patches before any model loading
    _patch_extractor_precision()
    if os.environ.get('RARAMURI_DISABLE_WHISPERX', '0') == '1' or transcript_backend == 'disabled':
        try:
            import pandas as pd
            import tribev2.eventstransforms as eventstransforms

            def empty_transcript(_wav_filename, _language):
                logger.info('Transcript backend disabled; returning empty transcript')
                return pd.DataFrame(columns=['text', 'start', 'duration', 'sequence_id', 'sentence'])

            eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(empty_transcript)
        except Exception as exc:
            logger.warning('Transcript disable patch failed: %s', exc)
    elif transcript_backend == 'parakeet':
        try:
            import pandas as pd
            import tribev2.eventstransforms as eventstransforms

            def parakeet_transcript_from_future(wav_filename, _language):
                """Wait on the pre-started Parakeet future, or run synchronously as fallback."""
                wav_key = str(wav_filename)
                cache_key, prepared_wav = _parakeet_audio_cache_key(wav_key)
                future = _PARAKEET_FUTURES.pop(cache_key, None)
                if future is None and len(_PARAKEET_FUTURES) == 1:
                    pending_key, future = _PARAKEET_FUTURES.popitem()
                    logger.info(
                        'Parakeet transcript: using sole pending future cache_key=%s for wav=%s cache_key=%s',
                        pending_key[:12],
                        wav_key,
                        cache_key[:12],
                    )
                if future is not None:
                    logger.info('Parakeet transcript: waiting on background future for %s', prepared_wav)
                    t0 = time.monotonic()
                    df = future.result(timeout=300)
                    logger.info('Parakeet transcript: future resolved in %.3fs words=%d', time.monotonic() - t0, len(df))
                    return df

                # Fallback: no pre-started future (e.g. chunked audio, second WAV).
                # Run Parakeet synchronously on CPU.
                logger.info('Parakeet transcript: no future for %s, running synchronously', prepared_wav)
                model_name = os.environ.get('RARAMURI_PARAKEET_MODEL', 'nvidia/parakeet-tdt-0.6b-v2')
                return _run_parakeet_cpu(prepared_wav, model_name)

            eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(parakeet_transcript_from_future)
            logger.info('Transcript backend set to Parakeet (early-parallel mode)')
        except Exception as exc:
            logger.warning('Parakeet transcript patch failed: %s', exc)

    try:
        _set_extractor_model_persistence(
            os.environ.get('RARAMURI_PERSIST_EXTRACTOR_MODELS', '0') == '1'
        )
    except Exception as exc:
        logger.warning('Extractor model persistence patch failed: %s', exc)

    if detailed_profile:
        try:
            import time as _time
            import neuralset as ns
            from neuralset.extractors.video import _HFVideoModel

            if not getattr(_HFVideoModel, '_raramuri_profiled', False):
                original_predict = _HFVideoModel.predict
                original_predict_hidden_states = _HFVideoModel.predict_hidden_states

                def profiled_predict(self, images, audio):
                    started = _time.monotonic()
                    out = original_predict(self, images, audio)
                    elapsed = _time.monotonic() - started
                    logger.info(
                        'VIDEO_MODEL predict seconds=%.3f model=%s images_shape=%s audio_present=%s',
                        elapsed,
                        getattr(self, 'model_name', 'unknown'),
                        getattr(images, 'shape', None),
                        audio is not None,
                    )
                    return out

                def profiled_predict_hidden_states(self, images, audio):
                    started = _time.monotonic()
                    out = original_predict_hidden_states(self, images, audio)
                    elapsed = _time.monotonic() - started
                    logger.info(
                        'VIDEO_MODEL predict_hidden_states seconds=%.3f model=%s images_shape=%s audio_present=%s output_shape=%s',
                        elapsed,
                        getattr(self, 'model_name', 'unknown'),
                        getattr(images, 'shape', None),
                        audio is not None,
                        getattr(out, 'shape', None),
                    )
                    return out

                _HFVideoModel.predict = profiled_predict
                _HFVideoModel.predict_hidden_states = profiled_predict_hidden_states
                _HFVideoModel._raramuri_profiled = True

            if not getattr(ns.extractors.base.BaseExtractor, '_raramuri_profiled', False):
                original_prepare = ns.extractors.base.BaseExtractor.prepare

                def profiled_prepare(self, events, *args, **kwargs):
                    started = _time.monotonic()
                    out = original_prepare(self, events, *args, **kwargs)
                    elapsed = _time.monotonic() - started
                    logger.info(
                        'EXTRACTOR prepare seconds=%.3f class=%s event_count=%s',
                        elapsed,
                        self.__class__.__name__,
                        len(events) if events is not None else None,
                    )
                    return out

                ns.extractors.base.BaseExtractor.prepare = profiled_prepare
                ns.extractors.base.BaseExtractor._raramuri_profiled = True

            logger.info('Detailed extractor profiling enabled')
        except Exception as exc:
            logger.warning('Detailed profiling patch failed: %s', exc)

    # ---------------------------------------------------------------------------
    # Parallel extractor preparation — must be before the video batching early-return
    # ---------------------------------------------------------------------------
    parallel_mode = os.environ.get('RARAMURI_PARALLEL_EXTRACTORS', '0').strip()
    if parallel_mode in ('1', '2'):
        try:
            import concurrent.futures
            import tribev2.main as tribev2_main

            # Model loading is serialized via _hf_load_lock (defined above in
            # _patch_extractor_precision) which wraps from_pretrained + .to(device).
            # Forwards are safe to overlap on the same CUDA device.

            if not getattr(tribev2_main.Data, '_raramuri_parallel_patched', False):
                _original_get_loaders = tribev2_main.Data.get_loaders

                def _parallel_get_loaders(self, events=None, split_to_build=None):
                    """Patched get_loaders that prepares extractors in parallel."""
                    import numpy as np
                    import pandas as pd
                    from neuralset.events.etypes import EventTypesHelper
                    from neuralset.events.utils import standardize_events

                    if events is None:
                        events = self.get_events()
                    else:
                        events = standardize_events(events)

                    # Build extractor dict (same logic as original)
                    extractors = {}
                    for modality in self.features_to_use:
                        extractors[modality] = getattr(self, f"{modality}_feature")
                    if "Fmri" in events.type.unique():
                        extractors["fmri"] = self.neuro
                    # Add dummy events (same as original)
                    dummy_events = []
                    for timeline_name, timeline in events.groupby("timeline"):
                        if "split" in timeline.columns:
                            splits = timeline.split.dropna().unique()
                            split = splits[0] if len(splits) == 1 else "all"
                        else:
                            split = "all"
                        dummy_events.append({
                            "type": "CategoricalEvent",
                            "timeline": timeline_name,
                            "start": timeline.start.min(),
                            "duration": timeline.stop.max() - timeline.start.min(),
                            "split": split,
                            "subject": timeline.subject.unique()[0],
                        })
                    events = pd.concat([events, pd.DataFrame(dummy_events)])
                    events = standardize_events(events)

                    extractors["subject_id"] = self.subject_id

                    # Remove extractors with no matching events
                    features_to_remove = set()
                    for extractor_name, extractor in extractors.items():
                        event_types = EventTypesHelper(extractor.event_types).names
                        if not any(et in events.type.unique() for et in event_types):
                            features_to_remove.add(extractor_name)
                    for name in features_to_remove:
                        del extractors[name]
                        logger.warning("Removing extractor %s (no corresponding events)", name)

                    # --- Parallel preparation ---
                    heavy = {}   # video — gets full GPU bandwidth
                    light = {}   # text, audio, subject_id — can overlap
                    for name, ext in extractors.items():
                        if name == 'video':
                            heavy[name] = ext
                        else:
                            light[name] = ext

                    def _prepare_one(name, ext):
                        t0 = time.monotonic()
                        logger.info("Preparing extractor (parallel): %s", name)
                        ext.prepare(events)
                        elapsed = time.monotonic() - t0
                        logger.info("Extractor %s prepared in %.3fs", name, elapsed)
                        if os.environ.get('RARAMURI_PERSIST_EXTRACTOR_MODELS', '0') != '1':
                            tribev2_main._free_extractor_model(ext)

                    if parallel_mode == '2':
                        # Aggressive: all extractors in parallel
                        logger.info("Parallel extractor prep (aggressive): %s",
                                    list(extractors.keys()))
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=len(extractors),
                            thread_name_prefix='extractor',
                        ) as pool:
                            futures = {
                                pool.submit(_prepare_one, name, ext): name
                                for name, ext in extractors.items()
                            }
                            for fut in concurrent.futures.as_completed(futures):
                                name = futures[fut]
                                exc = fut.exception()
                                if exc:
                                    logger.error("Extractor %s failed: %s", name, exc)
                                    raise exc
                    else:
                        # Mode 1: text+audio in parallel, then video
                        logger.info("Parallel extractor prep: light=%s then heavy=%s",
                                    list(light.keys()), list(heavy.keys()))
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=len(light),
                            thread_name_prefix='extractor',
                        ) as pool:
                            futures = {
                                pool.submit(_prepare_one, name, ext): name
                                for name, ext in light.items()
                            }
                            for fut in concurrent.futures.as_completed(futures):
                                name = futures[fut]
                                exc = fut.exception()
                                if exc:
                                    logger.error("Extractor %s failed: %s", name, exc)
                                    raise exc
                        for name, ext in heavy.items():
                            _prepare_one(name, ext)

                    # --- Build dataloaders (same as original) ---
                    import neuralset as ns
                    loaders = {}
                    if split_to_build is None:
                        splits_to_build = ["train", "val"]
                    else:
                        splits_to_build = [split_to_build]

                    for split in splits_to_build:
                        logger.info("Building dataloader for split %s", split)
                        if split == "all" or self.split_segments_by_time:
                            split_sel = [True] * len(events)
                            shuffle = False
                            overlap_trs = self.overlap_trs_train
                        else:
                            split_sel = events.split == split
                            if split not in events.split.unique():
                                shuffle = False
                            else:
                                shuffle = (
                                    self.shuffle_train if split == "train" else self.shuffle_val
                                )
                            if split == "val":
                                overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                            else:
                                overlap_trs = self.overlap_trs_train

                        sel = np.array(split_sel)
                        segments = ns.segments.list_segments(
                            events[sel],
                            triggers=events[sel].type == "CategoricalEvent",
                            stride=(self.duration_trs - overlap_trs) * self.TR,
                            duration=self.duration_trs * self.TR,
                            stride_drop_incomplete=self.stride_drop_incomplete,
                        )
                        if self.split_segments_by_time:
                            from tribev2.utils import split_segments_by_time
                            segments = split_segments_by_time(
                                segments,
                                val_ratio=self.study.transforms["split"].val_ratio,
                                split=split,
                            )
                        if len(segments) == 0:
                            logger.warning("No events found for split %s", split)
                            continue
                        dataset = ns.dataloader.SegmentDataset(
                            extractors=extractors,
                            segments=segments,
                            remove_incomplete_segments=False,
                        )
                        dataloader = dataset.build_dataloader(
                            shuffle=shuffle,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                        )
                        loaders[split] = dataloader
                    return loaders

                tribev2_main.Data.get_loaders = _parallel_get_loaders
                tribev2_main.Data._raramuri_parallel_patched = True
                logger.info('Patched Data.get_loaders for parallel extractor prep (mode=%s)', parallel_mode)
        except Exception as exc:
            logger.warning('Parallel extractor patch failed: %s', exc)

    if os.environ.get('RARAMURI_ENABLE_EXPERIMENTAL_VIDEO_BATCHING', '0') != '1':
        logger.info('Experimental video batching patch disabled')
    else:
        # --- Multi-clip VJEPA2 batching ---
        # Instead of processing 1 clip per forward pass, batch N clips together.
        # On GPUs with enough SMs (B200: ~160 SMs), a single ViT-G clip may not
        # saturate compute, so batching improves throughput.
        clip_batch_size = int(os.environ.get('RARAMURI_VJEPA_CLIP_BATCH_SIZE', '4'))
        try:
            import neuralset.extractors.video as video_module

            if not getattr(video_module.HuggingFaceVideo, '_raramuri_batched_patched', False):
                _original_get_data = video_module.HuggingFaceVideo._get_data

                def _batched_get_data(self, events):
                    """Patched _get_data that batches multiple clips per VJEPA2 forward."""
                    import numpy as np
                    import torch as _torch
                    import neuralset.base as nsbase
                    from neuralset.extractors.video import _HFVideoModel, _VideoImage, _fix_pixel_values

                    if not any(z in self.image.model_name for z in _HFVideoModel.MODELS):
                        yield from self._get_data_from_image_model(events)
                        return
                    if 'vjepa2' not in self.image.model_name:
                        # Only batch VJEPA2 — other models untested
                        yield from _original_get_data(self, events)
                        return

                    model = _HFVideoModel(
                        model_name=self.image.model_name,
                        pretrained=self.image.pretrained,
                        layer_type=self.layer_type,
                        num_frames=self.num_frames,
                    )
                    if model.model.device.type == "cpu":
                        model.model.to(self.image.device)

                    freq = events[0].frequency if self.frequency == "native" else self.frequency
                    T = 1 / freq if self.clip_duration is None else self.clip_duration
                    subtimes = list(k / model.num_frames * T for k in reversed(range(model.num_frames)))

                    for event in events:
                        video = event.read()
                        audio = video.audio if self.use_audio else None
                        freq_val = self.frequency if self.frequency != "native" else event.frequency
                        expect_frames = nsbase.Frequency(freq_val).to_ind(event.duration)
                        times = np.linspace(0, video.duration, expect_frames + 1)[1:]
                        output = np.array([])

                        # Collect all clips first
                        all_clips_data = []
                        all_clips_audio = []
                        for t in times:
                            ims = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
                            audio_clip = audio.subclipped(max(0, t - T), t) if audio is not None else None
                            pil_imgs = [i.read() for i in ims]
                            if pil_imgs and self.max_imsize is not None:
                                factor = max(pil_imgs[0].size) / self.max_imsize
                                if factor > 1:
                                    size = tuple(int(s / factor) for s in pil_imgs[0].size)
                                    pil_imgs = [pi.resize(size) for pi in pil_imgs]
                            data = np.array([np.array(pi) for pi in pil_imgs])
                            all_clips_data.append(data)
                            all_clips_audio.append(audio_clip)

                        # Process in batches
                        all_embeddings = []
                        for batch_start in range(0, len(all_clips_data), clip_batch_size):
                            batch_end = min(batch_start + clip_batch_size, len(all_clips_data))
                            batch_clips = all_clips_data[batch_start:batch_end]
                            B = len(batch_clips)

                            # Build batched input: pass each clip as a separate "video"
                            kwargs = {"videos": [list(clip) for clip in batch_clips], "return_tensors": "pt"}
                            inputs = model.processor(**kwargs)
                            _fix_pixel_values(inputs)
                            inputs = inputs.to(model.model.device)

                            t0_batch = time.monotonic()
                            with _torch.no_grad():
                                pred = model.model(**inputs)

                            elapsed_batch = time.monotonic() - t0_batch
                            logger.info(
                                'VJEPA2_BATCH clips=%d/%d batch_seconds=%.3f per_clip=%.3f',
                                B, len(all_clips_data), elapsed_batch, elapsed_batch / B,
                            )

                            states = pred.hidden_states
                            # states: tuple of (B, ...) tensors, one per layer
                            batched_out = _torch.cat([x.unsqueeze(1) for x in states], axis=1)
                            # batched_out: (B, L, ...)

                            for i in range(B):
                                t_embd = batched_out[i:i+1]  # keep batch dim for compatibility
                                t_embd = t_embd[0]  # remove batch dim
                                embd = self.image._aggregate_tokens(t_embd).cpu().numpy()
                                if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                                    embd = self.image._aggregate_layers(embd)
                                all_embeddings.append(embd)

                        for k, embd in enumerate(all_embeddings):
                            if not output.size:
                                output = np.zeros((len(times),) + embd.shape)
                                logger.debug("Created Tensor with size %s", output.shape)
                            output[k] = embd

                        video.close()
                        output = output.transpose(list(range(1, output.ndim)) + [0])
                        yield nsbase.TimedArray(
                            data=output.astype(np.float32),
                            frequency=freq_val,
                            start=nsbase._UNSET_START,
                            duration=event.duration,
                        )

                video_module.HuggingFaceVideo._get_data = _batched_get_data
                video_module.HuggingFaceVideo._raramuri_batched_patched = True
                logger.info('Patched VJEPA2 for multi-clip batching: clip_batch_size=%d', clip_batch_size)
        except Exception as exc:
            logger.warning('VJEPA2 batching patch failed: %s', exc)

    # --- torch.compile for VJEPA2 ---
    if os.environ.get('RARAMURI_VJEPA_COMPILE', '0') == '1':
        compile_mode = os.environ.get('RARAMURI_VJEPA_COMPILE_MODE', 'reduce-overhead')
        try:
            from neuralset.extractors.video import _HFVideoModel
            if not getattr(_HFVideoModel, '_raramuri_compile_patched', False):
                _original_compile_init = _HFVideoModel.__init__

                def _compile_video_init(self, *args, **kwargs):
                    _original_compile_init(self, *args, **kwargs)
                    if 'vjepa2' in self.model_name:
                        import torch as _torch
                        t0 = time.monotonic()
                        self.model = _torch.compile(self.model, mode=compile_mode)
                        logger.info('VJEPA2 torch.compile applied: mode=%s setup=%.3fs',
                                    compile_mode, time.monotonic() - t0)

                _HFVideoModel.__init__ = _compile_video_init
                _HFVideoModel._raramuri_compile_patched = True
                logger.info('Patched VJEPA2 for torch.compile: mode=%s', compile_mode)
        except Exception as exc:
            logger.warning('VJEPA2 compile patch failed: %s', exc)


def maybe_init_wandb():
    """Initialize W&B only when explicitly configured at runtime."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        return None

    try:
        import wandb
        run = wandb.init(project="raramuri-local", job_type="inference", reinit=True)
        globals()["_WANDB_RUN"] = run
        logger.info("W&B enabled: run_id=%s", getattr(run, "id", "unknown"))
        return run
    except Exception as exc:
        logger.warning("W&B init failed: %s", exc)
        return None


def load_model(reuse: bool = True):
    global _CACHED_MODEL
    import torch

    if reuse and _CACHED_MODEL is not None:
        logger.info('Reusing cached TRIBE model')
        return _CACHED_MODEL, {"model_load": 0.0, "model_reused": True}

    phase_timings = {}
    t0 = time.monotonic()
    with profiled_phase('model_load'):
        from tribev2 import TribeModel

        cache_dir = Path(os.environ.get("TRIBE_CACHE", "./cache"))
        cache_dir.mkdir(exist_ok=True)

        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(cache_dir))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_override = os.environ.get("RARAMURI_BATCH_SIZE")
        if batch_override:
            model.data.batch_size = int(batch_override)
        elif vram_gb >= 40:
            model.data.batch_size = 8
        elif vram_gb >= 20:
            model.data.batch_size = 4
        else:
            model.data.batch_size = 1

        data_workers_override = os.environ.get("RARAMURI_NUM_WORKERS")
        if data_workers_override:
            model.data.num_workers = int(data_workers_override)
        elif os.environ.get("RARAMURI_TRANSCRIPT_BACKEND", "").strip().lower() == "disabled" or os.environ.get("RARAMURI_DISABLE_WHISPERX", "0") == "1":
            model.data.num_workers = 0

        text_batch_override = os.environ.get("RARAMURI_TEXT_BATCH_SIZE")
        if text_batch_override and hasattr(model.data, 'text_feature'):
            model.data.text_feature.batch_size = int(text_batch_override)
        elif hasattr(model.data, 'text_feature') and model.data.text_feature.batch_size < 32:
            # Default batch_size=4 is unnecessarily small; process all words in
            # fewer forwards.  On a 130 GB GB10 the memory overhead is negligible
            # and 1 forward of 32 words is ~5× faster than 8 forwards of 4.
            model.data.text_feature.batch_size = 32
            logger.info("Text batch size raised to 32 (override with RARAMURI_TEXT_BATCH_SIZE)")

        image_batch_override = os.environ.get("RARAMURI_IMAGE_BATCH_SIZE")
        if image_batch_override and hasattr(model.data, 'video_feature') and hasattr(model.data.video_feature, 'image'):
            model.data.video_feature.image.batch_size = int(image_batch_override)

    phase_timings["model_load"] = round(time.monotonic() - t0, 3)
    logger.info("Model loaded: %.1fs (batch_size=%d num_workers=%d)", phase_timings["model_load"], model.data.batch_size, getattr(model.data, 'num_workers', -1))
    if reuse:
        _CACHED_MODEL = model
    return model, phase_timings


def log_runtime_contract() -> None:
    """Emit the image contract and runtime provenance expected by operators."""
    logger.info(
        "Image contract: authority=%s targets=%s base=%s python=%s torch_channel=%s",
        os.environ.get("RARAMURI_IMAGE_CONTRACT", "unknown"),
        os.environ.get("RARAMURI_IMAGE_TARGETS", "unknown"),
        os.environ.get("RARAMURI_IMAGE_BASE", "unknown"),
        os.environ.get("RARAMURI_IMAGE_PYTHON", "unknown"),
        os.environ.get("RARAMURI_IMAGE_TORCH_CHANNEL", "unknown"),
    )
    for key, expected in REQUIRED_ENV_DEFAULTS.items():
        actual = os.environ.get(key, expected)
        logger.info("Runtime path: %s=%s", key, actual)
        if actual != expected:
            logger.warning("Non-canonical %s override detected: expected %s", key, expected)


def _copy_missing_tree(source: Path, target: Path) -> None:
    """Copy only missing seed-cache entries into the runtime cache."""
    if not source.exists():
        return

    for root, dirnames, filenames in os.walk(source):
        root_path = Path(root)
        rel_path = root_path.relative_to(source)
        target_root = target / rel_path
        target_root.mkdir(parents=True, exist_ok=True)

        for dirname in dirnames:
            (target_root / dirname).mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            src_path = root_path / filename
            dst_path = target_root / filename
            if dst_path.exists():
                continue
            if src_path.is_symlink():
                dst_path.symlink_to(os.readlink(src_path))
            else:
                shutil.copy2(src_path, dst_path)


def hydrate_seed_caches() -> None:
    """Populate runtime caches from the image seed cache when available."""
    seed_root = Path(os.environ.get("RARAMURI_SEED_CACHE_ROOT", "/opt/raramuri-seed"))
    if not seed_root.exists():
        return

    seed_mappings = {
        "MNE_DATA": seed_root / "mne_data",
    }
    for key, source in seed_mappings.items():
        target = Path(os.environ[key])
        target.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            continue
        manifest = source.parent / "manifest.json"
        marker = target / ".raramuri-seed-manifest.json"
        seed_manifest = manifest.read_text() if manifest.exists() else ""
        current_manifest = marker.read_text() if marker.exists() else ""
        if seed_manifest and seed_manifest == current_manifest:
            continue
        logger.info("Hydrating runtime cache: %s <- %s", target, source)
        _copy_missing_tree(source, target)
        if seed_manifest:
            marker.write_text(seed_manifest)


def validate_runtime_assets() -> None:
    """Fail fast if required runtime assets are not available."""
    import spacy

    for model_name in ("en_core_web_sm", "en_core_web_lg"):
        try:
            spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError(f"Missing required spaCy model {model_name}") from exc


def warm_runtime_model_dependencies() -> None:
    """Instantiate model dependencies at container startup."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    hf_home = Path(os.environ["HF_HOME"])
    hf_home.mkdir(parents=True, exist_ok=True)
    token = ensure_hf_token_env()

    gated_model = "meta-llama/Llama-3.2-3B"
    if not token:
        raise RuntimeError(f"HF token env is required to warm {gated_model}")

    for model_name in ("facebook/w2v-bert-2.0", gated_model):
        started = time.monotonic()
        AutoModel.from_pretrained(model_name, cache_dir=str(hf_home), token=token)
        logger.info("Runtime HF model warmed: %s in %.1fs", model_name, time.monotonic() - started)

    vjepa_name = "facebook/vjepa2-vitg-fpc64-256"
    started = time.monotonic()
    try:
        AutoModel.from_pretrained(vjepa_name, cache_dir=str(hf_home), token=token)
        logger.info("Runtime HF model warmed: %s in %.1fs", vjepa_name, time.monotonic() - started)
    except Exception:
        snapshot_download(vjepa_name, cache_dir=str(hf_home), token=token)
        logger.info("Runtime HF snapshot warmed: %s in %.1fs", vjepa_name, time.monotonic() - started)


def ensure_runtime_prerequisites() -> None:
    """Fail clearly when the image contract is not actually available at runtime."""
    missing_tools = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing_tools:
        raise RuntimeError(f"Missing required runtime tools from image contract: {', '.join(missing_tools)}")

    for key, default in {**REQUIRED_ENV_DEFAULTS, "MNE_DATA": "/models/mne_data"}.items():
        _normalize_runtime_cache_path(key, default)
    ensure_hf_token_env()
    os.environ.setdefault("RARAMURI_SEED_CACHE_ROOT", "/opt/raramuri-seed")
    hydrate_seed_caches()
    validate_runtime_assets()


def _is_remote_video_source(source: str) -> bool:
    return "://" in source


def _coerce_time_value(value: str | float | int | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return str(value)

    text = str(value).strip()
    if not text:
        return None
    if text.startswith("-"):
        raise ValueError(f"{field_name} must be non-negative")
    return text


def _time_to_seconds(value: str | None) -> float | None:
    if value is None:
        return None

    parts = value.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        total = 0.0
        for part in parts:
            total = total * 60 + float(part)
        return total
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {value}") from exc


def normalize_time_range(
    start_time: str | float | int | None,
    end_time: str | float | int | None,
) -> tuple[str | None, str | None]:
    normalized_start = _coerce_time_value(start_time, field_name="start_time")
    normalized_end = _coerce_time_value(end_time, field_name="end_time")

    start_seconds = _time_to_seconds(normalized_start)
    end_seconds = _time_to_seconds(normalized_end)
    if (
        start_seconds is not None
        and end_seconds is not None
        and end_seconds <= start_seconds
    ):
        raise ValueError("end_time must be greater than start_time")

    return normalized_start, normalized_end


def clip_video(
    input_path: Path,
    output_path: Path,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> Path:
    """Clip a local video to the requested range using ffmpeg."""
    command = ["ffmpeg", "-y"]
    if start_time is not None:
        command.extend(["-ss", start_time])
    if end_time is not None:
        command.extend(["-to", end_time])
    command.extend(["-i", str(input_path), "-c", "copy", str(output_path)])
    subprocess.run(command, check=True, capture_output=True)
    logger.info(
        "Video clipped: input=%s output=%s start=%s end=%s",
        input_path,
        output_path,
        start_time,
        end_time,
    )
    return output_path


def download_video(
    url: str,
    output_path: Path,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> Path:
    """Download video from YouTube or another remote source using yt-dlp."""
    logger.info("Downloading video: %s", url)
    t0 = time.monotonic()
    command = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
    ]
    if start_time is not None or end_time is not None:
        section_start = start_time or "0"
        section_end = end_time or "inf"
        command.extend(["--download-sections", f"*{section_start}-{section_end}"])
        logger.info(
            "Downloading remote video section: start=%s end=%s",
            start_time,
            end_time,
        )
    command.append(url)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
    logger.info("Download complete: %.1fs", time.monotonic() - t0)
    return output_path


def prepare_video_input(
    source: str,
    *,
    start_time: str | float | int | None = None,
    end_time: str | float | int | None = None,
) -> tuple[Path, dict, Path | None]:
    """Materialize a local input video path for inference.

    Returns `(prepared_path, transfer_metadata, cleanup_dir)`.
    `cleanup_dir` is non-null when a temporary download/clip directory was created.
    """
    normalized_start, normalized_end = normalize_time_range(start_time, end_time)
    range_requested = normalized_start is not None or normalized_end is not None

    if _is_remote_video_source(source):
        cleanup_dir = Path(tempfile.mkdtemp(prefix="raramuri-input-"))
        output_path = cleanup_dir / "video.mp4"
        download_started_at = time.monotonic()
        download_video(
            source,
            output_path,
            start_time=normalized_start,
            end_time=normalized_end,
        )
        return output_path, {
            "seconds": round(time.monotonic() - download_started_at, 3),
            "source": "remote-download",
            "url": source,
            "start_time": normalized_start,
            "end_time": normalized_end,
        }, cleanup_dir

    local_file = Path(source).expanduser()
    if not local_file.exists() or not local_file.is_file():
        raise RuntimeError(f"Local input video does not exist: {local_file}")

    if not range_requested:
        return local_file, {
            "seconds": 0.0,
            "source": "local-file",
            "path": str(local_file),
        }, None

    cleanup_dir = Path(tempfile.mkdtemp(prefix="raramuri-input-"))
    output_path = cleanup_dir / f"{local_file.stem}-clip{local_file.suffix or '.mp4'}"
    clip_started_at = time.monotonic()
    clip_video(
        local_file,
        output_path,
        start_time=normalized_start,
        end_time=normalized_end,
    )
    return output_path, {
        "seconds": round(time.monotonic() - clip_started_at, 3),
        "source": "local-clip",
        "path": str(local_file),
        "start_time": normalized_start,
        "end_time": normalized_end,
    }, cleanup_dir


def extract_audio_wav(video_path: str) -> str:
    """Extract audio as 16kHz mono WAV."""
    video = Path(video_path).expanduser()
    stat = video.stat()
    cache_dir = Path(os.environ.get("RARAMURI_AUDIO_CACHE_DIR", "/tmp/raramuri-audio"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = f"{video.resolve(strict=False)}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    wav_path = cache_dir / f"{video.stem}-{digest}.wav"
    if wav_path.exists():
        logger.info("Audio extracted cache hit: %s", wav_path)
        return str(wav_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Audio extracted: %s", wav_path)
    return str(wav_path)


def compute_mel_spectrogram(wav_path: str, n_timesteps: int, n_mels: int = 128):
    """Compute mel spectrogram aligned to inference timesteps."""
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import spectrogram

    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / 32768.0

    total_samples = len(data)
    hop_length = max(1, total_samples // n_timesteps)
    nperseg = hop_length * 4

    freqs, times, Sxx = spectrogram(
        data, fs=rate, nperseg=min(nperseg, len(data)),
        noverlap=min(nperseg, len(data)) - hop_length,
        mode="magnitude",
    )

    # Triangular mel filterbank
    f_max = rate / 2
    mel_min = 2595 * np.log10(1 + 0 / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    freq_bins = np.linspace(0, f_max, len(freqs))

    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        for k in range(len(freqs)):
            if hz_points[m] <= freq_bins[k] <= hz_points[m + 1]:
                fb[m, k] = (freq_bins[k] - hz_points[m]) / (hz_points[m + 1] - hz_points[m] + 1e-10)
            elif hz_points[m + 1] < freq_bins[k] <= hz_points[m + 2]:
                fb[m, k] = (hz_points[m + 2] - freq_bins[k]) / (hz_points[m + 2] - hz_points[m + 1] + 1e-10)

    mel_spec = fb @ Sxx
    mel_spec = np.log1p(mel_spec * 100)

    # Resample to n_timesteps
    if mel_spec.shape[1] != n_timesteps:
        from scipy.ndimage import zoom
        mel_spec = zoom(mel_spec, (1, n_timesteps / mel_spec.shape[1]), order=1)

    mel_spec = mel_spec[:, :n_timesteps].T  # (n_timesteps, n_mels)
    if mel_spec.shape[0] < n_timesteps:
        pad = np.zeros((n_timesteps - mel_spec.shape[0], n_mels), dtype=np.float32)
        mel_spec = np.vstack([mel_spec, pad])

    return mel_spec.astype(np.float32)


def extract_captions(events, group_size=5):
    """Extract word-level captions from events DataFrame."""
    if "Word" not in events.type.values:
        return []

    word_events = events[events.type == "Word"].sort_values("start")
    words = []
    for _, row in word_events.iterrows():
        words.append({
            "word": str(row.get("text", row.get("filepath", ""))).strip(),
            "start": float(row["start"]),
            "end": float(row["start"] + row.get("duration", 0)),
        })

    # Group into caption phrases
    captions = []
    for i in range(0, len(words), group_size):
        group = words[i:i + group_size]
        if not group:
            continue
        text = " ".join(w["word"] for w in group if w["word"])
        if text.strip():
            captions.append({
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": text.strip(),
            })

    return captions


def inspect_text_embedding_status(model, events, audio_only: bool) -> dict:
    """Return text-embedding availability metadata and fail on silent mismatch.

    The acceptable cases are:
    - audio-only inference, where text is intentionally skipped
    - no Word events detected, where there is nothing to embed
    - a configured text extractor/model is present

    If Word events exist but the model has no text extractor configured, raise a
    clear error instead of silently omitting text embeddings.
    """
    model_data = getattr(model, "data", None)
    text_feature = getattr(model_data, "text_feature", None)
    extractor_present = text_feature is not None
    extractor_class = None if text_feature is None else text_feature.__class__.__name__
    model_name = None if text_feature is None else getattr(text_feature, "model_name", None)

    if audio_only:
        status = "audio_only"
        word_event_count = 0
    else:
        word_event_count = int((events.type == "Word").sum()) if "type" in events.columns else 0
        if word_event_count == 0:
            status = "no_word_events"
        elif extractor_present and model_name:
            status = "available"
        else:
            raise RuntimeError(
                "Word events were detected, but the text embedding extractor/model is unavailable "
                "(missing model.data.text_feature or model_name). Refusing to continue silently."
            )

    metadata = {
        "status": status,
        "word_event_count": word_event_count,
        "extractor_present": extractor_present,
        "extractor_class": extractor_class,
        "model_name": model_name,
    }
    logger.info("Text embedding status: %s", metadata)
    return metadata


def _emit_progress(
    progress_callback: Callable[[dict], None] | None,
    *,
    step: str,
    stage: str,
    step_index: int,
    total_steps: int,
    inference_started_at: float,
    step_started_at: float | None = None,
    extra: dict | None = None,
) -> None:
    if progress_callback is None:
        return

    completed_steps = step_index if stage == 'completed' else max(step_index - 1, 0)
    payload = {
        'step': step,
        'stage': stage,
        'step_index': step_index,
        'total_steps': total_steps,
        'completed_steps': completed_steps,
        'progress_ratio': round(completed_steps / total_steps, 4) if total_steps else 0.0,
        'elapsed_seconds': round(time.monotonic() - inference_started_at, 3),
        'step_elapsed_seconds': None if step_started_at is None else round(time.monotonic() - step_started_at, 3),
        'runtime': runtime_status_snapshot(),
    }
    if extra:
        payload.update(extra)

    try:
        progress_callback(payload)
    except Exception as exc:
        logger.warning("Progress callback failed for step=%s stage=%s: %s", step, stage, exc)


def run_inference(video_path: str, progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Run TRIBE v2 inference and return full results bundle."""
    import numpy as np
    import torch

    # --- Wipe exca feature cache if requested ---
    # This ensures timed runs measure actual extraction, not cache reads.
    if os.environ.get('RARAMURI_WIPE_FEATURE_CACHE', '0') == '1':
        import shutil
        cache_root = Path(os.environ.get("TRIBE_CACHE", "./cache"))
        wiped = 0
        for d in cache_root.glob("neuralset.extractors.*"):
            if d.is_dir():
                shutil.rmtree(d)
                wiped += 1
        if wiped:
            logger.info('Wiped %d exca feature cache directories under %s', wiped, cache_root)

    # Enable TF32 matmul — free performance on Ampere+ with no quality impact
    # on the final predictions (verified: NRMSE=0% for BF16, the default path).
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Kick off early Parakeet transcription on CPU (before anything GPU) ---
    # This extracts audio WAV (fast, ~0.1s via ffmpeg) and submits NeMo
    # transcription to a background thread.  Runs fully parallel with GPU
    # model load + event construction below.
    start_early_parakeet(video_path)

    patch_runtime_extractors()

    inference_started_at = time.monotonic()
    phase_timings = {}
    warmup_predict = os.environ.get("RARAMURI_WARMUP_PREDICT", "0") == "1"
    skip_metrics = os.environ.get("RARAMURI_SKIP_METRICS", "0") == "1"
    phase_order = ["model_load", "event_build", "caption_extract"]
    if warmup_predict:
        phase_order.append("predict_warmup")
    phase_order.extend(["predict", "spectrum", "metrics"])
    total_steps = len(phase_order)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this image contract requires a working NVIDIA runtime")

    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Torch runtime: version=%s cuda=%s", torch.__version__, torch.version.cuda)

    # On unified-memory hardware (DGX Spark / GB10) GPU and system RAM are the
    # same physical pool.  Without a cap, model loading can exhaust the pool and
    # OOM-kill the container.  70% leaves headroom for the OS and CPU-side work
    # (NeMo/Parakeet runs on CPU).  Override via RARAMURI_VRAM_FRACTION if a
    # different deployment target needs a different ceiling.
    _vram_fraction = float(os.environ.get("RARAMURI_VRAM_FRACTION", "0.70"))
    torch.cuda.set_per_process_memory_fraction(_vram_fraction)
    logger.info("VRAM fraction cap: %.0f%%", _vram_fraction * 100)

    # --- Load model ---
    step_index = phase_order.index("model_load") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="model_load",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
    )
    model, model_phase_timings = load_model(reuse=True)
    phase_timings.update(model_phase_timings)
    _emit_progress(
        progress_callback,
        step="model_load",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"timing_seconds": phase_timings.get("model_load")},
    )

    # --- Build events (full pipeline unless profiling opts out) ---
    t0 = time.monotonic()
    audio_only = os.environ.get("RARAMURI_AUDIO_ONLY", "0") == "1"
    step_index = phase_order.index("event_build") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="event_build",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"audio_only": audio_only},
    )
    with profiled_phase('event_build'):
        if audio_only:
            import pandas as pd
            from tribev2.demo_utils import get_audio_and_text_events
            event = {
                "type": "Video",
                "filepath": str(video_path),
                "start": 0,
                "timeline": "default",
                "subject": "default",
            }
            events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        else:
            events = model.get_events_dataframe(video_path=video_path)
    phase_timings["event_build"] = round(time.monotonic() - t0, 3)
    logger.info("Events built (%s): %.1fs", "audio-only" if audio_only else "full pipeline", phase_timings["event_build"])
    has_text = not audio_only
    _emit_progress(
        progress_callback,
        step="event_build",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"timing_seconds": phase_timings["event_build"]},
    )

    event_summary = events.type.value_counts().to_dict()
    logger.info("Event types: %s", event_summary)
    text_embedding_status = inspect_text_embedding_status(model, events, audio_only)

    # --- Extract captions from Word events ---
    t0 = time.monotonic()
    step_index = phase_order.index("caption_extract") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="caption_extract",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"has_text": has_text},
    )
    with profiled_phase('caption_extract'):
        captions = [] if audio_only else extract_captions(events)
    phase_timings["caption_extract"] = round(time.monotonic() - t0, 3)
    logger.info("Captions extracted: %d phrases (has_text=%s)", len(captions), has_text)
    _emit_progress(
        progress_callback,
        step="caption_extract",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={
            "timing_seconds": phase_timings["caption_extract"],
            "caption_count": len(captions),
            "text_embedding_status": text_embedding_status["status"],
        },
    )

    # --- Run prediction ---
    if warmup_predict:
        tw = time.monotonic()
        step_index = phase_order.index("predict_warmup") + 1
        step_started_at = time.monotonic()
        _emit_progress(
            progress_callback,
            step="predict_warmup",
            stage="started",
            step_index=step_index,
            total_steps=total_steps,
            inference_started_at=inference_started_at,
            step_started_at=step_started_at,
        )
        with profiled_phase('predict_warmup'):
            warmup_preds, warmup_segments = model.predict(events=events, verbose=True)
            logger.info(
                'Warmup prediction complete: preds_shape=%s kept_segments=%d',
                getattr(warmup_preds, 'shape', None),
                len(warmup_segments),
            )
        phase_timings["predict_warmup"] = round(time.monotonic() - tw, 3)
        _emit_progress(
            progress_callback,
            step="predict_warmup",
            stage="completed",
            step_index=step_index,
            total_steps=total_steps,
            inference_started_at=inference_started_at,
            step_started_at=step_started_at,
            extra={"timing_seconds": phase_timings["predict_warmup"]},
        )

    t0 = time.monotonic()
    step_index = phase_order.index("predict") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="predict",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
    )
    try:
        with profiled_phase('predict'):
            preds, segments = model.predict(events=events, verbose=True)
            preds = preds.astype(np.float32)
    except RuntimeError as exc:
        # Some long-lived server sessions can hit a stateful freeze/unfreeze
        # mismatch when extractor persistence is enabled. Recover by disabling
        # persistence and rebuilding a fresh model once.
        if (
            'Cannot unfreeze partially without first freezing the module with `freeze()`' in str(exc)
            and os.environ.get('RARAMURI_PERSIST_EXTRACTOR_MODELS', '0') == '1'
        ):
            logger.warning(
                'Predict failed with extractor persistence enabled (%s). '
                'Retrying once with persistence disabled and fresh model cache.',
                exc,
            )
            os.environ['RARAMURI_PERSIST_EXTRACTOR_MODELS'] = '0'
            patch_runtime_extractors()
            global _CACHED_MODEL
            _CACHED_MODEL = None
            model, model_phase_timings_retry = load_model(reuse=True)
            if model_phase_timings_retry.get('model_load'):
                phase_timings['model_load_retry'] = model_phase_timings_retry['model_load']
            with profiled_phase('predict_retry_no_persist'):
                preds, segments = model.predict(events=events, verbose=True)
                preds = preds.astype(np.float32)
        else:
            raise
    elapsed = time.monotonic() - t0
    phase_timings["predict"] = round(elapsed, 3)
    logger.info("Prediction complete: %.1fs — shape %s", elapsed, preds.shape)
    _emit_progress(
        progress_callback,
        step="predict",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"timing_seconds": phase_timings["predict"], "prediction_shape": list(preds.shape)},
    )

    # --- Compute spectrum ---
    t0 = time.monotonic()
    spectrum = None
    spectrum_shape = [0, 0]
    step_index = phase_order.index("spectrum") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="spectrum",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
    )
    try:
        wav_path = extract_audio_wav(video_path)
        mel = compute_mel_spectrogram(wav_path, preds.shape[0])
        spectrum = mel.tolist()
        spectrum_shape = list(mel.shape)
        Path(wav_path).unlink(missing_ok=True)
        phase_timings["spectrum"] = round(time.monotonic() - t0, 3)
        logger.info("Spectrum computed: %s in %.1fs", spectrum_shape, phase_timings["spectrum"])
    except Exception as e:
        phase_timings["spectrum"] = round(time.monotonic() - t0, 3)
        logger.warning("Spectrum computation failed: %s", e)
    _emit_progress(
        progress_callback,
        step="spectrum",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"timing_seconds": phase_timings["spectrum"], "spectrum_shape": spectrum_shape},
    )

    # --- Compute metrics ---
    t0 = time.monotonic()
    step_index = phase_order.index("metrics") + 1
    step_started_at = time.monotonic()
    _emit_progress(
        progress_callback,
        step="metrics",
        stage="started",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"skip_metrics": skip_metrics},
    )
    if skip_metrics:
        phase_timings["metrics"] = round(time.monotonic() - t0, 3)
        logger.info("Metrics skipped for profiling")
        segment_times = [
            {"start": float(s.start), "duration": float(s.duration)}
            for s in segments
        ]
        result = {
            "predictions": preds.tolist(),
            "shape": list(preds.shape),
            "segments": segment_times,
            "event_types": event_summary,
            "has_text": has_text,
            "text_embeddings": {
                **text_embedding_status,
                "caption_count": len(captions),
            },
            "captions": captions,
            "spectrum": spectrum,
            "spectrum_shape": spectrum_shape,
            "timing": {
                "total_seconds": round(time.monotonic() - inference_started_at, 3),
                "phases": phase_timings,
            },
            "metrics": {
                "n_timesteps": int(preds.shape[0]),
                "n_vertices": int(preds.shape[1]),
                "skipped": True,
            },
            "network_names": [],
            "network_colors": [],
            "roi_to_network": {},
        }
        _emit_progress(
            progress_callback,
            step="metrics",
            stage="completed",
            step_index=step_index,
            total_steps=total_steps,
            inference_started_at=inference_started_at,
            step_started_at=step_started_at,
            extra={"timing_seconds": phase_timings["metrics"], "metrics_skipped": True},
        )
        _emit_progress(
            progress_callback,
            step="complete",
            stage="completed",
            step_index=total_steps,
            total_steps=total_steps,
            inference_started_at=inference_started_at,
            extra={"result_shape": list(preds.shape)},
        )
        return result

    from tribev2.utils import get_hcp_labels, get_hcp_vertex_labels

    mesh = "fsaverage5"
    roi_labels = get_hcp_labels(mesh=mesh, combine=False, hemi="both")
    vertex_labels = get_hcp_vertex_labels(mesh=mesh, combine=False)
    roi_names = list(roi_labels.keys())

    network_names = [
        "Visual", "Somatomotor", "Dorsal Attention",
        "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode",
    ]
    network_colors = [
        "#7B287D", "#4C68B1", "#00A074", "#C43AFA",
        "#E8DC7A", "#E69422", "#CD3E4E",
    ]

    _HCP_TO_YEO = {
        "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V3A": 0, "V3B": 0,
        "V6": 0, "V6A": 0, "V7": 0, "V8": 0, "V3CD": 0,
        "LO1": 0, "LO2": 0, "LO3": 0, "PIT": 0, "VVC": 0,
        "VMV1": 0, "VMV2": 0, "VMV3": 0, "V4t": 0, "FST": 0,
        "FFC": 0, "PHA1": 0, "PHA2": 0, "PHA3": 0,
        "MT": 0, "MST": 0, "PH": 0, "DVT": 0, "ProS": 0,
        "4": 1, "3a": 1, "3b": 1, "1": 1, "2": 1,
        "6d": 1, "6mp": 1, "6ma": 1, "6v": 1, "6r": 1,
        "SCEF": 1, "FEF": 1, "PEF": 1, "43": 1,
        "OP4": 1, "OP1": 1, "OP2-3": 1, "RI": 1,
        "LBelt": 1, "MBelt": 1, "PBelt": 1,
        "A1": 1, "A4": 1, "A5": 1, "TA2": 1,
        "52": 1, "RetI": 1, "Ig": 1, "5m": 1, "5L": 1, "5mv": 1,
        "7AL": 2, "7Am": 2, "7PL": 2, "7Pm": 2, "7PC": 2,
        "LIPv": 2, "LIPd": 2, "VIP": 2, "MIP": 2, "AIP": 2,
        "IPS1": 2, "IP0": 2, "IP1": 2, "IP2": 2, "PFt": 2,
        "FOP1": 3, "FOP2": 3, "FOP3": 3, "FOP4": 3, "FOP5": 3,
        "PFcm": 3, "PFop": 3, "PF": 3, "PFm": 3, "MI": 3,
        "44": 3, "45": 3, "IFJa": 3, "IFJp": 3,
        "IFSa": 3, "IFSp": 3, "47l": 3,
        "6a": 3, "i6-8": 3, "s6-8": 3,
        "STV": 3, "TPOJ1": 3, "TPOJ2": 3, "TPOJ3": 3,
        "PSL": 3, "Peri": 3, "PI": 3, "A32pr": 3, "p24pr": 3,
        "EC": 4, "PreS": 4, "H": 4, "TF": 4,
        "TGd": 4, "TGv": 4, "TE2a": 4, "TE2p": 4,
        "10pp": 4, "10r": 4, "OFC": 4, "pOFC": 4,
        "25": 4, "s32": 4, "a24": 4, "Aol": 4, "13l": 4,
        "47s": 4, "47m": 4, "STGa": 4,
        "p9-46v": 5, "a9-46v": 5, "46": 5, "9-46d": 5,
        "9a": 5, "9p": 5, "9m": 5,
        "8Av": 5, "8Ad": 5, "8BL": 5, "8C": 5,
        "p47r": 5, "a47r": 5, "10v": 5, "10d": 5,
        "a10p": 5, "p10p": 5, "11l": 5, "a24pr": 5,
        "d32": 5, "p32": 5, "p32pr": 5,
        "PGs": 5, "PGi": 5, "PGp": 5,
        "TE1a": 5, "TE1m": 5, "TE1p": 5,
        "STSva": 5, "STSvp": 5, "STSda": 5, "STSdp": 5,
        "RSC": 6, "POS1": 6, "POS2": 6,
        "v23ab": 6, "d23ab": 6, "23c": 6, "23d": 6,
        "31a": 6, "31pv": 6, "31pd": 6,
        "7m": 6, "PCV": 6, "PrCv": 6, "a32": 6,
        "33pr": 6, "24dd": 6, "24dv": 6, "p24": 6,
        "55b": 6, "SFL": 6, "8BM": 6,
        "Pol1": 6, "Pol2": 6, "PHT": 6, "TempPar": 6, "PreCu": 6,
    }

    def _clean_roi(name):
        return name.replace("-lh", "").replace("-rh", "").replace("_ROI", "").strip()

    n_verts = preds.shape[1]
    network_map = np.zeros(n_verts, dtype=np.uint8)
    for i, label in enumerate(vertex_labels[:n_verts]):
        clean = _clean_roi(label)
        if clean in _HCP_TO_YEO:
            network_map[i] = _HCP_TO_YEO[clean]

    roi_to_network = {name: _HCP_TO_YEO.get(_clean_roi(name), 0) for name in roi_names}

    roi_metrics = {}
    for name, verts in roi_labels.items():
        verts = verts[verts < n_verts]
        if len(verts) == 0:
            continue
        ts = np.mean(preds[:, verts], axis=1).astype(np.float64)
        grad = np.gradient(ts)
        roi_metrics[name] = {
            "mean": float(np.mean(ts)),
            "std": float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
            "peak": float(np.max(ts)),
            "range": float(np.ptp(ts)),
            "gradient_mean": float(np.mean(np.abs(grad))),
            "network": roi_to_network.get(name, 0),
            "timeseries": ts.tolist(),
        }

    roi_ranked = sorted(
        roi_metrics.keys(), key=lambda k: abs(roi_metrics[k]["mean"]), reverse=True
    )

    global_mean = np.mean(preds, axis=1).tolist()

    spatial_entropy = []
    for t in range(preds.shape[0]):
        vals = preds[t]
        shifted = vals - vals.min() + 1e-10
        p = shifted / shifted.sum()
        ent = -np.sum(p * np.log2(p))
        max_ent = np.log2(len(p))
        spatial_entropy.append(float(ent / max_ent) if max_ent > 0 else 0.0)

    T = preds.shape[0]
    net_activations = np.zeros((T, 7))
    for net_idx in range(7):
        mask = network_map == net_idx
        if mask.any():
            net_activations[:, net_idx] = np.mean(preds[:, mask], axis=1)
    dominant = np.argmax(net_activations, axis=1).tolist()

    top_k = 30
    n_rois = len(roi_names)
    roi_ts = np.zeros((n_rois, T))
    for i, name in enumerate(roi_names):
        verts = roi_labels[name]
        verts = verts[verts < n_verts]
        if len(verts) > 0:
            roi_ts[i] = np.mean(preds[:, verts], axis=1)
    mean_act = np.mean(np.abs(roi_ts), axis=1)
    top_indices = np.argsort(mean_act)[-top_k:][::-1]
    top_names = [roi_names[i] for i in top_indices]
    top_ts_data = roi_ts[top_indices]
    if T >= 3:
        corr = np.corrcoef(top_ts_data)
        corr = np.nan_to_num(corr, nan=0.0)
    else:
        corr = np.eye(len(top_indices))

    logger.info("Metrics computed: %.1fs", time.monotonic() - t0)
    phase_timings["metrics"] = round(time.monotonic() - t0, 3)
    _emit_progress(
        progress_callback,
        step="metrics",
        stage="completed",
        step_index=step_index,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        step_started_at=step_started_at,
        extra={"timing_seconds": phase_timings["metrics"], "metrics_skipped": False},
    )

    segment_times = [
        {"start": float(s.start), "duration": float(s.duration)}
        for s in segments
    ]

    result = {
        "predictions": preds.tolist(),
        "shape": list(preds.shape),
        "segments": segment_times,
        "event_types": event_summary,
        "has_text": has_text,
        "text_embeddings": {
            **text_embedding_status,
            "caption_count": len(captions),
        },
        "captions": captions,
        "spectrum": spectrum,
        "spectrum_shape": spectrum_shape,
        "timing": {
            "total_seconds": round(time.monotonic() - inference_started_at, 3),
            "phases": phase_timings,
        },
        "metrics": {
            "roi": roi_metrics,
            "roi_ranked": roi_ranked,
            "global": {
                "mean": global_mean,
                "spatial_entropy": spatial_entropy,
                "network_dominance": {
                    "dominant": dominant,
                    "activations": net_activations.tolist(),
                },
                "correlation": {
                    "names": top_names,
                    "matrix": corr.tolist(),
                },
            },
            "n_timesteps": T,
            "n_vertices": int(n_verts),
        },
        "network_names": network_names,
        "network_colors": network_colors,
        "roi_to_network": roi_to_network,
    }

    _emit_progress(
        progress_callback,
        step="complete",
        stage="completed",
        step_index=total_steps,
        total_steps=total_steps,
        inference_started_at=inference_started_at,
        extra={"result_shape": list(preds.shape)},
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="TRIBE v2 cloud inference")
    parser.add_argument("url", help="YouTube video URL or local file path")
    parser.add_argument("--output", "-o", default="results.json", help="Output JSON path")
    parser.add_argument("--start-time", help="Optional clip start time in seconds or HH:MM:SS")
    parser.add_argument("--end-time", help="Optional clip end time in seconds or HH:MM:SS")
    args = parser.parse_args()

    overall_started_at = time.monotonic()
    transfer_timings = {}

    log_runtime_contract()
    ensure_runtime_prerequisites()
    maybe_init_wandb()

    logger.info("Input probe: raw=%s remote=%s", args.url, _is_remote_video_source(args.url))
    cleanup_dir = None
    video_path, transfer_timings["input_prepare"], cleanup_dir = prepare_video_input(
        args.url,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    result = run_inference(str(video_path))
    result.setdefault("timing", {})["transfer"] = transfer_timings

    output_started_at = time.monotonic()
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f)
    result["timing"]["output_write"] = {
        "seconds": round(time.monotonic() - output_started_at, 3),
        "path": str(output_path),
    }
    result["timing"]["wall_clock_total_seconds"] = round(time.monotonic() - overall_started_at, 3)
    with open(output_path, "w") as f:
        json.dump(result, f)
    logger.info("Results written to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    logger.info("Timing: wall_clock_total_seconds=%.3f", result["timing"]["wall_clock_total_seconds"])

    wandb_run = globals().get("_WANDB_RUN")
    if wandb_run is not None:
        try:
            wandb_run.summary.update({
                "timing/total_seconds": result["timing"].get("total_seconds"),
                "timing/wall_clock_total_seconds": result["timing"].get("wall_clock_total_seconds"),
                "result/n_timesteps": result.get("metrics", {}).get("n_timesteps"),
                "result/n_vertices": result.get("metrics", {}).get("n_vertices"),
                "result/has_text": result.get("has_text"),
            })
            wandb_run.finish()
        except Exception:
            pass

    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
