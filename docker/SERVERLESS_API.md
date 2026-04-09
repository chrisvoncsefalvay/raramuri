# Rarámuri Serverless API

## Endpoint

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

All requests require the header `Authorization: Bearer {RUNPOD_API_KEY}`.

## Request schema

```json
{
  "input": {
    "video_url": "https://www.youtube.com/watch?v=...",
    "start_time": "00:00:10",
    "end_time": "00:00:40",
    "include_predictions": true
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_url` | string | yes* | YouTube URL or direct video URL |
| `video_path` | string | yes* | Path on network volume (alternative to URL) |
| `start_time` | string | no | Clip start, e.g. `"00:00:10"` |
| `end_time` | string | no | Clip end, e.g. `"00:00:40"` |
| `include_predictions` | bool | no | Include raw predictions array (default `true`) |

\* One of `video_url` or `video_path` is required.

## Streaming progress

The handler is a streaming generator. While inference runs, it yields progress
updates, then emits the final result as the last item. There are three ways to
consume results:

### 1. Async with streaming (`/run` + `/stream`)

Submit the job, then poll `/stream/{job_id}` for incremental updates:

```python
import requests
import time

API_KEY = "your_runpod_api_key"
ENDPOINT = "your_endpoint_id"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Submit
resp = requests.post(f"{BASE}/run", headers=HEADERS, json={
    "input": {"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
              "start_time": "00:00:10", "end_time": "00:00:40"}
})
job_id = resp.json()["id"]

# Poll for streamed updates
while True:
    stream = requests.get(f"{BASE}/stream/{job_id}", headers=HEADERS).json()
    for item in stream.get("stream", []):
        if item["output"]["type"] == "progress":
            step = item["output"]["step"]
            ratio = item["output"].get("progress_ratio", 0)
            elapsed = item["output"].get("elapsed_seconds", 0)
            print(f"  [{ratio:.0%}] {step} ({elapsed:.1f}s)")
        elif item["output"]["type"] == "result":
            print("Done!", item["output"]["timing"]["total_seconds"], "s")
    if stream.get("status") in ("COMPLETED", "FAILED"):
        break
    time.sleep(2)
```

### 2. Async without streaming (`/run` + `/status`)

With `return_aggregate_stream: True`, the final `/status` response aggregates
all yielded values into the `output` field as a list. The last element is the
result:

```python
resp = requests.post(f"{BASE}/run", headers=HEADERS, json={
    "input": {"video_url": "https://example.com/video.mp4"}
})
job_id = resp.json()["id"]

while True:
    status = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS).json()
    if status["status"] in ("COMPLETED", "FAILED"):
        break
    time.sleep(5)

# output is a list: [progress, progress, ..., result]
result = status["output"][-1]  # last item is the full result
```

### 3. Synchronous (`/runsync`)

Blocks until completion (90s timeout — may not be enough for cold starts):

```python
resp = requests.post(f"{BASE}/runsync", headers=HEADERS, json={
    "input": {"video_url": "https://example.com/video.mp4",
              "include_predictions": False}
})
data = resp.json()
if data["status"] == "COMPLETED":
    result = data["output"][-1]
```

## Progress update schema

Each progress update has:

```json
{
  "type": "progress",
  "step": "feature_extraction",
  "stage": "started",
  "step_index": 3,
  "total_steps": 8,
  "progress_ratio": 0.25,
  "elapsed_seconds": 12.3,
  "step_elapsed_seconds": 1.2
}
```

Pipeline steps in order: `model_load`, `transcription`, `event_construction`,
`feature_extraction`, `prediction`, `metrics`, `correlation`, `roi`.

## Result schema

The final yield (with `"type": "result"`) contains:

| Field | Description |
|-------|-------------|
| `timing.phases` | Per-phase seconds |
| `timing.total_seconds` | Wall-clock total |
| `timing.warmup` | Model warm-up breakdown |
| `timing.warm_start` | `true` if models were already loaded |
| `timing.transfer` | Video download/prepare metadata |
| `captions` | Transcription segments (start, end, text) |
| `predictions` | Raw prediction array (omitted if `include_predictions=false`) |
| `segments` | Detected segments |
| `event_types` | Count of each event type |
| `metrics` | Computed metrics (correlation, ROI, etc.) |
| `metrics_text` | Prometheus text exposition format |

## Environment tuning

These env vars are set on the RunPod template for optimal A100 80GB performance:

| Variable | Value | Effect |
|----------|-------|--------|
| `RARAMURI_BATCH_SIZE` | `16` | TRIBEv2 batch size |
| `RARAMURI_ENABLE_VIDEO_BATCHING` | `1` | Batch VJEPA2 clips per forward |
| `RARAMURI_VJEPA_CLIP_BATCH_SIZE` | `8` | Clips per VJEPA2 batch |
| `RARAMURI_PARALLEL_EXTRACTORS` | `2` | Overlap extractors on CUDA streams |
| `RARAMURI_PERSIST_EXTRACTOR_MODELS` | `1` | Keep models in VRAM + torch.compile |
| `RARAMURI_VRAM_FRACTION` | `0.95` | VRAM cap (dedicated GPU, not unified) |
| `RARAMURI_TRANSCRIPT_BACKEND` | `parakeet` | Parakeet on CPU (faster than WhisperX) |
| `RARAMURI_VJEPA_QUANT` | `fp8` | FP8 quantization for video model |
| `RARAMURI_AUDIO_QUANT` | `fp8` | FP8 quantization for audio model |
| `RARAMURI_TEXT_QUANT` | `fp8` | FP8 quantization for text model |
