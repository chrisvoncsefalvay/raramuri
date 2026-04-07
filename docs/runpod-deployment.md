# RunPod Deployment Spec — Rarámuri

Two deployment modes: **GPU Pod** (persistent instance) and **Serverless** (scale-to-zero). Both consume the same OCI image from `ghcr.io/chrisvoncsefalvay/raramuri:latest`, but serverless requires an additional handler entrypoint.

---

## 1. GPU Pod Deployment

### What it is

A dedicated GPU VM running the container persistently. You pay per hour whether it's serving requests or idle. Equivalent to renting a cloud GPU box.

### When to use

- Development/testing
- Sustained workloads with consistent traffic
- When cold-start latency is unacceptable and you want the models always hot
- Interactive debugging or visualization workflows

### Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **GPU type** | A100 80GB / H100 80GB / A6000 48GB | Multiple large models loaded simultaneously; 48GB minimum, 80GB recommended |
| **Container image** | `ghcr.io/chrisvoncsefalvay/raramuri:latest` | Or private registry |
| **Container disk** | 50 GB | Image layers + temp video files |
| **Volume disk** | 100 GB | Mounted at `/models` — persists HF cache, TRIBE cache, MNE data across restarts |
| **Volume mount** | `/models` | Maps to `HF_HOME=/models/hf`, `TRIBE_CACHE=/models/tribe`, `MNE_DATA=/models/mne_data` |
| **Exposed ports** | `8765/http` | The inference server port |
| **Docker command** | `python /app/infer_server.py --port 8765` | Override default entrypoint (which runs CLI `infer.py`) |

### Environment variables

```
HF_TOKEN=hf_xxxxx                          # Required — Llama 3.2 3B is gated
HF_HOME=/models/hf                         # Already set in image
TRIBE_CACHE=/models/tribe                  # Already set in image
MNE_DATA=/models/mne_data                  # Already set in image
RARAMURI_SERVER_MAX_PENDING_REQUESTS=2     # Optional — backpressure tuning
RARAMURI_SERVER_REQUEST_TIMEOUT=900        # Optional — 15min default
```

### Startup sequence

1. Pod boots, pulls image (first run only — cached on volume disk after)
2. `infer_server.py` starts, calls `warm_runtime_model_dependencies()` which loads all 5 models into GPU memory
3. Server reports ready on `GET /ready` (returns 503 until warm)
4. First inference after warm-up is fast (~35s for a 30s clip on GB10)

### Access

- RunPod provides a public URL: `https://{pod_id}-8765.proxy.runpod.net`
- No built-in auth — you must add your own (reverse proxy, API key middleware, or restrict to RunPod's SSH tunnel)
- SSH access available for debugging

### Cost estimate

| GPU | $/hr (community) | $/hr (secure) |
|-----|-------------------|---------------|
| A6000 48GB | ~$0.40 | ~$0.76 |
| A100 80GB | ~$1.10 | ~$1.64 |
| H100 80GB | ~$2.50 | ~$3.89 |

Volume storage: $0.07/GB/month ($7/mo for 100GB).

---

## 2. Serverless Deployment

### What it is

RunPod manages a pool of workers that scale from 0 to N based on incoming requests. You pay per second of active compute. Workers cold-start when scaling up.

### When to use

- Production API with variable/bursty traffic
- Cost optimization when usage is sporadic
- Managed scaling without infrastructure ops

### Architecture difference

RunPod Serverless does **not** call your HTTP server. Instead, it imports a **handler function** from your code. You must add a `handler.py` entrypoint:

```python
# docker/handler.py — RunPod serverless entrypoint
import runpod
import json
import logging
import tempfile
import os

from infer import (
    ensure_runtime_prerequisites,
    load_model,
    log_runtime_contract,
    patch_runtime_extractors,
    prepare_video_input,
    run_inference,
    warm_runtime_model_dependencies,
)

logger = logging.getLogger(__name__)

# ── Model warm-up at import time ──────────────────────────────────
# RunPod calls the handler function per-request, but the module is
# loaded once per worker lifetime. Warming here means the model
# stays hot across requests on the same worker.
log_runtime_contract()
ensure_runtime_prerequisites()
patch_runtime_extractors()
load_model()
warm_runtime_model_dependencies()
logger.info("RunPod worker warm and ready")


def handler(job):
    """RunPod serverless handler.

    Input schema (job["input"]):
        video_url: str          — YouTube URL or direct video URL
        video_path: str         — path on network volume (alternative to URL)
        start_time: str | None  — e.g. "00:00:10"
        end_time: str | None    — e.g. "00:00:40"

    Returns:
        dict with predictions, segments, metrics
    """
    inp = job["input"]

    video_url = inp.get("video_url")
    video_path = inp.get("video_path")
    start_time = inp.get("start_time")
    end_time = inp.get("end_time")

    if not video_url and not video_path:
        return {"error": "Either video_url or video_path is required"}

    source = video_path or video_url

    try:
        result = run_inference(source)
        return result
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
```

### Dockerfile changes for serverless

Add the RunPod SDK and handler to the image:

```dockerfile
# Append to existing Dockerfile for serverless variant
RUN python -m pip install --no-cache-dir runpod

COPY docker/handler.py /app/handler.py

# Serverless entrypoint overrides the CLI default
# Set via RunPod template config, not baked into image:
#   CMD ["python", "-u", "/app/handler.py"]
```

Or create a separate `Dockerfile.serverless` that extends the base:

```dockerfile
FROM ghcr.io/chrisvoncsefalvay/raramuri:latest

RUN python -m pip install --no-cache-dir runpod
COPY docker/handler.py /app/handler.py

ENTRYPOINT []
CMD ["python", "-u", "/app/handler.py"]
```

### Serverless template configuration

| Setting | Value |
|---------|-------|
| **Container image** | Your serverless-variant image |
| **Container disk** | 50 GB |
| **GPU type** | A100 80GB / H100 80GB (48GB minimum) |
| **Max workers** | Start with 1–3, tune based on traffic |
| **Idle timeout** | 300s (5 min) — keeps warm worker alive between requests |
| **Execution timeout** | 600s (10 min) — enough for a 30s clip |
| **Network volume** | Mount at `/models` (see persistence section below) |

### Environment variables (set in template)

```
HF_TOKEN=hf_xxxxx
HF_HOME=/models/hf
TRIBE_CACHE=/models/tribe
MNE_DATA=/models/mne_data
```

### Serverless authentication

RunPod Serverless has **built-in API key authentication**. Every request must include:

```
Authorization: Bearer {RUNPOD_API_KEY}
```

The API key is generated per account at `https://www.runpod.io/console/user/settings` under "API Keys". You can create multiple keys with different permissions.

**Invocation URL format:**
```
POST https://api.runpod.ai/v2/{endpoint_id}/run        # async
POST https://api.runpod.ai/v2/{endpoint_id}/runsync     # sync (blocks until done)
GET  https://api.runpod.ai/v2/{endpoint_id}/status/{id} # poll async job
```

**Example call:**
```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "start_time": "00:00:00",
      "end_time": "00:00:30"
    }
  }'
```

**Webhook support:** For async jobs, you can pass a `webhook` URL in the request body to get a POST callback when the job completes:
```json
{
  "input": { "video_url": "..." },
  "webhook": "https://your-server.com/callback"
}
```

### Serverless persistence: Network Volumes

Cold starts for Rarámuri are **very expensive** (downloading 5 large models: VJEPA2, Wav2Vec, Llama 3.2, Parakeet, TRIBEv2 weights). Without persistence, every cold start re-downloads ~30GB+ of model weights.

**Network Volumes** solve this:

| Feature | Details |
|---------|---------|
| **What** | Persistent NFS storage attached to serverless workers |
| **Mount point** | `/models` (or `/runpod-volume`, configurable) |
| **Survives** | Worker scale-down, restarts, and redeployment |
| **Region-locked** | Volume must be in the same datacenter region as workers |
| **Pricing** | $0.07/GB/month |
| **Recommended size** | 100 GB (covers all model caches + headroom) |

**Setup steps:**

1. Create a network volume in your target region (e.g., `US-TX-3`)
2. Pre-populate it by running a GPU Pod with the volume attached, starting the server, and letting warm-up download all models
3. Reference the volume ID in your serverless template configuration
4. Set `HF_HOME=/models/hf` etc. so models resolve to the volume

**After pre-population**, cold starts only need to load models from the volume into GPU memory (~60–90s) rather than downloading them (~10–20 min).

### Cold start vs warm request latency

| Scenario | Estimated latency |
|----------|-------------------|
| **Warm worker** (models in GPU) | ~35s for 30s clip |
| **Cold start + network volume** (models on disk) | ~90–150s total (60–90s load + 35s inference) |
| **Cold start, no volume** (full download) | ~15–25 min (unacceptable for production) |

**Mitigation strategies:**
- Set `Idle Timeout` to 300–600s to keep at least one worker warm
- Set `Min Workers = 1` if you can afford always-on cost (~$1–3/hr depending on GPU)
- Use `Active Workers` (always-warm) instead of `Max Workers` (scale-to-zero) for latency-critical deployments — this is effectively GPU Pod pricing but with RunPod's managed scaling
- Network volume is **mandatory** for acceptable cold-start times

---

## 3. Comparison matrix

| Dimension | GPU Pod | Serverless |
|-----------|---------|------------|
| **Pricing model** | $/hr (always on) | $/sec (active compute only) |
| **Scaling** | Manual | Automatic 0→N workers |
| **Cold start** | N/A (always running) | 90–150s with network volume |
| **Auth** | None built-in (DIY) | Built-in API key |
| **API contract** | Your HTTP server (full REST) | RunPod handler (input→output) |
| **Visualization endpoint** | Yes (`/jobs/{id}/visualize`) | Not directly (handler returns data only) |
| **SSH access** | Yes | No |
| **Best for** | Dev/test, sustained load, interactive | Production API, variable traffic, cost optimization |
| **Model volume** | Optional but recommended | Effectively mandatory |
| **Max idle cost** | Full GPU rate 24/7 | $0 (scale to zero) or min-worker rate |

---

## 4. Recommended approach

**For production:** Serverless with a network volume + `Min Workers = 1`. You get managed scaling, built-in auth, and keep one worker warm to avoid cold starts on the critical path. Cost is comparable to a GPU Pod during active hours, but you scale to just one worker during quiet periods instead of paying for idle capacity.

**For development:** GPU Pod with a persistent volume. Direct SSH access, full REST API including visualization, and no handler refactoring needed.

---

## 5. Implementation checklist

- [ ] Create `docker/handler.py` (RunPod serverless handler)
- [ ] Create `docker/Dockerfile.serverless` extending base image
- [ ] Build and push serverless image to registry
- [ ] Create RunPod network volume (100GB) in target region
- [ ] Pre-populate volume via GPU Pod (run warm-up, download all models)
- [ ] Create serverless template with volume, env vars, GPU config
- [ ] Test `runsync` endpoint with a short clip
- [ ] Tune idle timeout and min/max workers based on traffic patterns
