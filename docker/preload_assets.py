#!/usr/bin/env python3
"""Preload non-token runtime assets into the image.

This script is intended for image builds. It fetches non-gated assets that do
not require a runtime Hugging Face token, leaving actual model instantiation to
container startup.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


SEED_ROOT = Path(os.environ.get("RARAMURI_SEED_CACHE_ROOT", "/opt/raramuri-seed"))
MNE_DATA = Path(os.environ.setdefault("MNE_DATA", str(SEED_ROOT / "mne_data")))
MANIFEST_PATH = SEED_ROOT / "manifest.json"


def _log(message: str) -> None:
    print(message, flush=True)


def _timed(label: str, fn):
    started = time.monotonic()
    result = fn()
    _log(f"{label}: {time.monotonic() - started:.1f}s")
    return result


def _preload_spacy() -> list[str]:
    import spacy

    loaded = []
    for model_name in ("en_core_web_sm", "en_core_web_lg"):
        _timed(f"spaCy load {model_name}", lambda name=model_name: spacy.load(name))
        loaded.append(model_name)
    return loaded


def _preload_nltk() -> list[str]:
    import nltk

    loaded = []
    for resource in ("punkt", "punkt_tab"):
        _timed(
            f"NLTK download {resource}",
            lambda name=resource: nltk.download(name, quiet=True, raise_on_error=True),
        )
        loaded.append(resource)
    return loaded


def _preload_mne() -> str:
    import mne

    mne.set_config("MNE_DATA", str(MNE_DATA))
    return str(_timed("MNE fsaverage", lambda: mne.datasets.fetch_fsaverage(verbose=True)))


def main() -> int:
    SEED_ROOT.mkdir(parents=True, exist_ok=True)
    MNE_DATA.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed_root": str(SEED_ROOT),
        "mne_data": str(MNE_DATA),
        "assets": {},
    }

    manifest["assets"]["spacy"] = _preload_spacy()
    manifest["assets"]["nltk"] = _preload_nltk()
    manifest["assets"]["mne_fsaverage"] = _preload_mne()

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _log(f"Wrote seed manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
