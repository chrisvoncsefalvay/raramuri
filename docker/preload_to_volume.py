#!/usr/bin/env python3
"""Download all required models to a RunPod network volume.

Run this ONCE on a cheap pod with the network volume mounted, then
every serverless worker reuses the pre-staged weights instantly.

Usage:
    # On a pod with volume at /runpod-volume:
    HF_TOKEN=hf_xxx python /app/preload_to_volume.py /runpod-volume

    # Custom model subset:
    HF_TOKEN=hf_xxx python /app/preload_to_volume.py /runpod-volume --skip-parakeet
"""

import argparse
import os
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, flush=True)


def _timed(label: str):
    """Context manager that logs elapsed time."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.monotonic()
            _log(f"Downloading {label}...")
            return self
        def __exit__(self, *exc):
            _log(f"  {label}: {time.monotonic() - self.t0:.1f}s")
    return _Timer()


def main():
    parser = argparse.ArgumentParser(description="Pre-stage models to a network volume")
    parser.add_argument("volume_path", help="Mount point of the RunPod network volume")
    parser.add_argument("--skip-parakeet", action="store_true", help="Skip Parakeet ASR model")
    args = parser.parse_args()

    vol = Path(args.volume_path)
    if not vol.is_dir():
        _log(f"FATAL: {vol} is not a directory")
        sys.exit(1)

    hf_dir = vol / "models" / "hf"
    tribe_dir = vol / "models" / "tribe"
    mne_dir = vol / "models" / "mne_data"
    for d in (hf_dir, tribe_dir, mne_dir):
        d.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        _log("FATAL: HF_TOKEN environment variable required")
        sys.exit(1)

    os.environ["HF_HOME"] = str(hf_dir)
    os.environ["TRIBE_CACHE"] = str(tribe_dir)
    os.environ["MNE_DATA"] = str(mne_dir)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    # HF gated/public models
    for name in ("facebook/w2v-bert-2.0", "meta-llama/Llama-3.2-3B"):
        with _timed(name):
            AutoModel.from_pretrained(name, cache_dir=str(hf_dir), token=token)

    # V-JEPA2
    vjepa = "facebook/vjepa2-vitg-fpc64-256"
    with _timed(vjepa):
        try:
            AutoModel.from_pretrained(vjepa, cache_dir=str(hf_dir), token=token)
        except Exception:
            snapshot_download(vjepa, cache_dir=str(hf_dir), token=token)

    # TRIBEv2
    with _timed("facebook/tribev2"):
        from tribev2 import TribeModel
        TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(tribe_dir))

    # Parakeet ASR
    if not args.skip_parakeet:
        with _timed("nvidia/parakeet-tdt-0.6b-v2"):
            import nemo.collections.asr as nemo_asr
            nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    _log(f"\nAll models staged to {vol}/models/")
    _log(f"  HF cache:    {hf_dir}")
    _log(f"  TRIBE cache: {tribe_dir}")
    _log(f"  MNE data:    {mne_dir}")

    # Write a marker so the handler can verify the volume is ready
    marker = vol / "models" / ".ready"
    marker.write_text("ok\n")
    _log(f"  Marker:      {marker}")


if __name__ == "__main__":
    main()
